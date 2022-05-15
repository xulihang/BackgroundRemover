import numpy as np
import cv2


def remove_background(back,text):

    fgbg = cv2.createBackgroundSubtractorMOG2()

    fgmask = fgbg.apply(back)
    fgmask = fgbg.apply(text)
    
    
    
    ret, thresh1 = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((4, 4), np.uint8)
    img_dilate = cv2.dilate(thresh1, kernel)
    img_erode = cv2.erode(img_dilate, kernel)

    contours, hierarchy = cv2.findContours(img_erode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    height = back.shape[0]
    width = back.shape[1]
    imgWhite = np.ones((width, height), np.uint8) * 255
    for bbox in bounding_boxes:
        [x , y, w, h] = bbox
        thresh = binarize_box(back,text,bbox)
        imgWhite[y:y+h, x:x+w] = thresh
    cv2.imshow('frame',thresh1)
    cv2.imshow('text',text)
    cv2.imshow('imgWhite',imgWhite)
    cv2.imwrite('fgmask.jpg',text)
    cv2.imwrite('out.jpg',imgWhite)
    k = cv2.waitKey()
    cv2.destroyAllWindows()


def draw_bounding_box(src,bounding_boxes):
    for bbox in bounding_boxes:
        [x , y, w, h] = bbox
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
def binarize_box(bg,text,bbox):
    [x , y, w, h] = bbox
    bg_crop = bg[y:y+h, x:x+w]
    text_crop = text[y:y+h, x:x+w]
    rough_remove_background_with_original_image(bg_crop,text_crop)
    gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
    
    
    ret, thresh1 = cv2.threshold(gray, 200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if ret<150:
        ret, thresh1 = cv2.threshold(gray, ret+100,255,cv2.THRESH_BINARY_INV)
    print(ret)
    return thresh1

def rough_remove_background_with_original_image(bg,text):
    for y in range(0,len(bg)):
        for x in range(0,len(bg[y])):
            if similar_pixel(bg[y][x],text[y][x]):
                text[y][x] = [0,0,0]
    
def similar_pixel(point1,point2):
    diff = 0
    diff = diff + abs(point1[0] - point2[0])
    diff = diff + abs(point1[1] - point2[1])
    diff = diff + abs(point1[2] - point2[2])
    if diff > 10:
        return False
    else:
        return True
        
if __name__ == "__main__" :

    back = cv2.imread("337_B_002.jpg")
    #back = cv2.imread("213_0101.jpg")

    text = cv2.imread("007_A_003.jpg")
    #text = cv2.imread("003_0101.jpg")
    remove_background(back,text)
