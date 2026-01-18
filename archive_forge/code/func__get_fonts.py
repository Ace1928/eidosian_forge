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
def _get_fonts(self) -> Tuple[Set[str], Set[str]]:
    """
        Get the names of embedded fonts and unembedded fonts.

        Returns:
            A tuple (Set of embedded fonts, set of unembedded fonts)
        """
    obj = self.get_object()
    assert isinstance(obj, DictionaryObject)
    fonts: Set[str] = set()
    embedded: Set[str] = set()
    fonts, embedded = _get_fonts_walk(obj, fonts, embedded)
    unembedded = fonts - embedded
    return (embedded, unembedded)