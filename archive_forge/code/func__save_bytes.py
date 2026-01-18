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
def _save_bytes(self, image: LTImage) -> str:
    """Save an image without encoding, just bytes"""
    name, path = self._create_unique_image_name(image, '.jpg')
    width, height = image.srcsize
    channels = len(image.stream.get_data()) / width / height / (image.bits / 8)
    with open(path, 'wb') as fp:
        try:
            from PIL import Image
            from PIL import ImageOps
        except ImportError:
            raise ImportError(PIL_ERROR_MESSAGE)
        mode: Literal['1', 'L', 'RGB', 'CMYK']
        if image.bits == 1:
            mode = '1'
        elif image.bits == 8 and channels == 1:
            mode = 'L'
        elif image.bits == 8 and channels == 3:
            mode = 'RGB'
        elif image.bits == 8 and channels == 4:
            mode = 'CMYK'
        img = Image.frombytes(mode, image.srcsize, image.stream.get_data(), 'raw')
        if mode == 'L':
            img = ImageOps.invert(img)
        img.save(fp)
    return name