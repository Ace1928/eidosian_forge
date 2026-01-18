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
def _layout_mode_fonts(self) -> Dict[str, _layout_mode.Font]:
    """
        Get fonts formatted for "layout" mode text extraction.

        Returns:
            Dict[str, Font]: dictionary of _layout_mode.Font instances keyed by font name
        """
    objr: Any = self
    fonts: Dict[str, _layout_mode.Font] = {}
    while objr is not None:
        try:
            resources_dict: Any = objr[PG.RESOURCES]
        except KeyError:
            resources_dict = {}
        if '/Font' in resources_dict and self.pdf is not None:
            for font_name in resources_dict['/Font']:
                *cmap, font_dict_obj = build_char_map(font_name, 200.0, self)
                font_dict = {k: v.get_object() if isinstance(v, IndirectObject) else [_v.get_object() for _v in v] if isinstance(v, ArrayObject) else v for k, v in font_dict_obj.items()}
                fonts[font_name] = _layout_mode.Font(*cmap, font_dict)
        try:
            objr = objr['/Parent'].get_object()
        except KeyError:
            objr = None
    return fonts