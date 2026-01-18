import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _img_to_ascii(self, path):
    im = Image.open(path)
    im.thumbnail((60, 40), Image.BICUBIC)
    im = im.convert('L')
    asc = []
    for y in range(0, im.size[1]):
        for x in range(0, im.size[0]):
            lum = 255 - im.getpixel((x, y))
            asc.append(_greyscale[lum * len(_greyscale) // 256])
        asc.append('\n')
    return ''.join(asc)