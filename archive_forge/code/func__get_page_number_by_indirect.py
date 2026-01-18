import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def _get_page_number_by_indirect(self, indirect_reference: Union[None, int, NullObject, IndirectObject]) -> Optional[int]:
    """
        Generate _page_id2num.

        Args:
            indirect_reference:

        Returns:
            The page number or None
        """
    if indirect_reference is None or isinstance(indirect_reference, NullObject):
        return None
    if isinstance(indirect_reference, int):
        indirect_reference = IndirectObject(indirect_reference, 0, self)
    obj = indirect_reference.get_object()
    if isinstance(obj, PageObject):
        return obj.page_number
    return None