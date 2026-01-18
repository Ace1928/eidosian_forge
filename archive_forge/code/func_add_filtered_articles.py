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
def add_filtered_articles(self, fltr: Union[Pattern[Any], str], pages: Dict[int, PageObject], reader: PdfReader) -> None:
    """
        Add articles matching the defined criteria.

        Args:
            fltr:
            pages:
            reader:
        """
    if isinstance(fltr, str):
        fltr = re.compile(fltr)
    elif not isinstance(fltr, Pattern):
        fltr = re.compile('')
    for p in pages.values():
        pp = p.original_page
        for a in pp.get('/B', ()):
            thr = a.get_object().get('/T')
            if thr is None:
                continue
            else:
                thr = thr.get_object()
            if thr.indirect_reference.idnum not in self._id_translated[id(reader)] and fltr.search((thr['/I'] if '/I' in thr else {}).get('/Title', '')):
                self._add_articles_thread(thr, pages, reader)