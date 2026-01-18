import logging
import struct
import sys
from io import BytesIO
from typing import (
from . import settings
from .cmapdb import CMap
from .cmapdb import CMapBase
from .cmapdb import CMapDB
from .cmapdb import CMapParser
from .cmapdb import FileUnicodeMap
from .cmapdb import IdentityUnicodeMap
from .cmapdb import UnicodeMap
from .encodingdb import EncodingDB
from .encodingdb import name2unicode
from .fontmetrics import FONT_METRICS
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import num_value
from .pdftypes import resolve1, resolve_all
from .pdftypes import stream_value
from .psparser import KWD
from .psparser import LIT
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import literal_name
from .utils import Matrix, Point
from .utils import Rect
from .utils import apply_matrix_norm
from .utils import choplist
from .utils import nunpack
class Type1FontHeaderParser(PSStackParser[int]):
    KEYWORD_BEGIN = KWD(b'begin')
    KEYWORD_END = KWD(b'end')
    KEYWORD_DEF = KWD(b'def')
    KEYWORD_PUT = KWD(b'put')
    KEYWORD_DICT = KWD(b'dict')
    KEYWORD_ARRAY = KWD(b'array')
    KEYWORD_READONLY = KWD(b'readonly')
    KEYWORD_FOR = KWD(b'for')

    def __init__(self, data: BinaryIO) -> None:
        PSStackParser.__init__(self, data)
        self._cid2unicode: Dict[int, str] = {}
        return

    def get_encoding(self) -> Dict[int, str]:
        """Parse the font encoding.

        The Type1 font encoding maps character codes to character names. These
        character names could either be standard Adobe glyph names, or
        character names associated with custom CharStrings for this font. A
        CharString is a sequence of operations that describe how the character
        should be drawn. Currently, this function returns '' (empty string)
        for character names that are associated with a CharStrings.

        Reference: Adobe Systems Incorporated, Adobe Type 1 Font Format

        :returns mapping of character identifiers (cid's) to unicode characters
        """
        while 1:
            try:
                cid, name = self.nextobject()
            except PSEOF:
                break
            try:
                self._cid2unicode[cid] = name2unicode(cast(str, name))
            except KeyError as e:
                log.debug(str(e))
        return self._cid2unicode

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_PUT:
            (_, key), (_, value) = self.pop(2)
            if isinstance(key, int) and isinstance(value, PSLiteral):
                self.add_results((key, literal_name(value)))
        return