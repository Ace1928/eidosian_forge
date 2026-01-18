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
def _create_unique_image_name(self, image: LTImage, ext: str) -> Tuple[str, str]:
    name = image.name + ext
    path = os.path.join(self.outdir, name)
    img_index = 0
    while os.path.exists(path):
        name = '%s.%d%s' % (image.name, img_index, ext)
        path = os.path.join(self.outdir, name)
        img_index += 1
    return (name, path)