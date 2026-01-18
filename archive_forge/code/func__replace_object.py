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
def _replace_object(self, indirect_reference: Union[int, IndirectObject], obj: PdfObject) -> PdfObject:
    if isinstance(indirect_reference, IndirectObject):
        if indirect_reference.pdf != self:
            raise ValueError('pdf must be self')
        indirect_reference = indirect_reference.idnum
    gen = self._objects[indirect_reference - 1].indirect_reference.generation
    if getattr(obj, 'indirect_reference', None) is not None and obj.indirect_reference.pdf != self:
        obj = obj.clone(self)
    self._objects[indirect_reference - 1] = obj
    obj.indirect_reference = IndirectObject(indirect_reference, gen, self)
    return self._objects[indirect_reference - 1]