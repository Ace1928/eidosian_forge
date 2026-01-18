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
def _save_raw(self, image: LTImage) -> str:
    """Save an image with unknown encoding"""
    ext = '.%d.%dx%d.img' % (image.bits, image.srcsize[0], image.srcsize[1])
    name, path = self._create_unique_image_name(image, ext)
    with open(path, 'wb') as fp:
        fp.write(image.stream.get_data())
    return name