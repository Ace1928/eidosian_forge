import os
import os.path
import struct
from io import BytesIO
from typing import BinaryIO, Tuple
from .jbig2 import JBIG2StreamReader, JBIG2StreamWriter
from .layout import LTImage
from .pdfcolor import LITERAL_DEVICE_CMYK
from .pdfcolor import LITERAL_DEVICE_GRAY
from .pdfcolor import LITERAL_DEVICE_RGB
from .pdftypes import (
def _save_bmp(self, image: LTImage, width: int, height: int, bytes_per_line: int, bits: int) -> str:
    """Save a BMP encoded image"""
    name, path = self._create_unique_image_name(image, '.bmp')
    with open(path, 'wb') as fp:
        bmp = BMPWriter(fp, bits, width, height)
        data = image.stream.get_data()
        i = 0
        for y in range(height):
            bmp.write_line(y, data[i:i + bytes_per_line])
            i += bytes_per_line
    return name