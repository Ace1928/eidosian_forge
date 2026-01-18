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
def _get_contents_as_bytes(self) -> Optional[bytes]:
    """
        Return the page contents as bytes.

        Returns:
            The ``/Contents`` object as bytes, or ``None`` if it doesn't exist.

        """
    if PG.CONTENTS in self:
        obj = self[PG.CONTENTS].get_object()
        if isinstance(obj, list):
            return b''.join((x.get_object().get_data() for x in obj))
        else:
            return cast(bytes, cast(EncodedStreamObject, obj).get_data())
    else:
        return None