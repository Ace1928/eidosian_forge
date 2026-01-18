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
def _set_page_label(self, page_index_from: int, page_index_to: int, style: Optional[PageLabelStyle]=None, prefix: Optional[str]=None, start: Optional[int]=0) -> None:
    """
        Set a page label to a range of pages.

        Page indexes must be given
        starting from 0. Labels must have a style, a prefix or both. If to a
        range is not assigned any page label a decimal label starting from 1 is
        applied.

        Args:
            page_index_from: page index of the beginning of the range starting from 0
            page_index_to: page index of the beginning of the range starting from 0
            style:  The numbering style to be used for the numeric portion of each page label:
                        /D Decimal arabic numerals
                        /R Uppercase roman numerals
                        /r Lowercase roman numerals
                        /A Uppercase letters (A to Z for the first 26 pages,
                           AA to ZZ for the next 26, and so on)
                        /a Lowercase letters (a to z for the first 26 pages,
                           aa to zz for the next 26, and so on)
            prefix: The label prefix for page labels in this range.
            start:  The value of the numeric portion for the first page label
                    in the range.
                    Subsequent pages are numbered sequentially from this value,
                    which must be greater than or equal to 1. Default value: 1.
        """
    default_page_label = DictionaryObject()
    default_page_label[NameObject('/S')] = NameObject('/D')
    new_page_label = DictionaryObject()
    if style is not None:
        new_page_label[NameObject('/S')] = NameObject(style)
    if prefix is not None:
        new_page_label[NameObject('/P')] = TextStringObject(prefix)
    if start != 0:
        new_page_label[NameObject('/St')] = NumberObject(start)
    if NameObject(CatalogDictionary.PAGE_LABELS) not in self._root_object:
        nums = ArrayObject()
        nums_insert(NumberObject(0), default_page_label, nums)
        page_labels = TreeObject()
        page_labels[NameObject('/Nums')] = nums
        self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)] = page_labels
    page_labels = cast(TreeObject, self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)])
    nums = cast(ArrayObject, page_labels[NameObject('/Nums')])
    nums_insert(NumberObject(page_index_from), new_page_label, nums)
    nums_clear_range(NumberObject(page_index_from), page_index_to, nums)
    next_label_pos, *_ = nums_next(NumberObject(page_index_from), nums)
    if next_label_pos != page_index_to + 1 and page_index_to + 1 < len(self.pages):
        nums_insert(NumberObject(page_index_to + 1), default_page_label, nums)
    page_labels[NameObject('/Nums')] = nums
    self._root_object[NameObject(CatalogDictionary.PAGE_LABELS)] = page_labels