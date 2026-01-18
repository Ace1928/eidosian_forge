import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
@property
def _old_images(self) -> List[File]:
    """
        Get a list of all images of the page.

        This requires pillow. You can install it via 'pip install pypdf[image]'.

        For the moment, this does NOT include inline images. They will be added
        in future.
        """
    images_extracted: List[File] = []
    if RES.XOBJECT not in self[PG.RESOURCES]:
        return images_extracted
    x_object = self[PG.RESOURCES][RES.XOBJECT].get_object()
    for obj in x_object:
        if x_object[obj][IA.SUBTYPE] == '/Image':
            extension, byte_stream, img = _xobj_to_image(x_object[obj])
            if extension is not None:
                filename = f'{obj[1:]}{extension}'
                images_extracted.append(File(name=filename, data=byte_stream))
                images_extracted[-1].image = img
                images_extracted[-1].indirect_reference = x_object[obj].indirect_reference
    return images_extracted