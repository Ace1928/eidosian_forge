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
def _save_jpeg2000(self, image: LTImage) -> str:
    """Save a JPEG 2000 encoded image"""
    raw_data = image.stream.get_rawdata()
    assert raw_data is not None
    name, path = self._create_unique_image_name(image, '.jp2')
    with open(path, 'wb') as fp:
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(PIL_ERROR_MESSAGE)
        ifp = BytesIO(raw_data)
        i = Image.open(ifp)
        i.save(fp, 'JPEG2000')
    return name