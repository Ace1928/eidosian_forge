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
def _clone_outline(self, dest: Destination) -> TreeObject:
    n_ol = TreeObject()
    self._add_object(n_ol)
    n_ol[NameObject('/Title')] = TextStringObject(dest['/Title'])
    if not isinstance(dest['/Page'], NullObject):
        if dest.node is not None and '/A' in dest.node:
            n_ol[NameObject('/A')] = dest.node['/A'].clone(self)
        else:
            n_ol[NameObject('/Dest')] = dest.dest_array
    if dest.node is not None:
        n_ol[NameObject('/F')] = NumberObject(dest.node.get('/F', 0))
        n_ol[NameObject('/C')] = ArrayObject(dest.node.get('/C', [FloatObject(0.0), FloatObject(0.0), FloatObject(0.0)]))
    return n_ol