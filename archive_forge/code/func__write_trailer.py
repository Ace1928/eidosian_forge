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
def _write_trailer(self, stream: StreamType, xref_location: int) -> None:
    """
        Write the PDF trailer to the stream.

        To quote the PDF specification:
            [The] trailer [gives] the location of the cross-reference table and
            of certain special objects within the body of the file.
        """
    stream.write(b'trailer\n')
    trailer = DictionaryObject()
    trailer.update({NameObject(TK.SIZE): NumberObject(len(self._objects) + 1), NameObject(TK.ROOT): self._root, NameObject(TK.INFO): self._info_obj})
    if self._ID:
        trailer[NameObject(TK.ID)] = self._ID
    if self._encrypt_entry:
        trailer[NameObject(TK.ENCRYPT)] = self._encrypt_entry.indirect_reference
    trailer.write_to_stream(stream)
    stream.write(f'\nstartxref\n{xref_location}\n%%EOF\n'.encode())