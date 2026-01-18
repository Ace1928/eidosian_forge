import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
def _getobj_parse(self, pos: int, objid: int) -> object:
    assert self._parser is not None
    self._parser.seek(pos)
    _, objid1 = self._parser.nexttoken()
    _, genno = self._parser.nexttoken()
    _, kwd = self._parser.nexttoken()
    if objid1 != objid:
        x = []
        while kwd is not self.KEYWORD_OBJ:
            _, kwd = self._parser.nexttoken()
            x.append(kwd)
        if len(x) >= 2:
            objid1 = x[-2]
    if objid1 != objid:
        raise PDFSyntaxError('objid mismatch: {!r}={!r}'.format(objid1, objid))
    if kwd != KWD(b'obj'):
        raise PDFSyntaxError('Invalid object spec: offset=%r' % pos)
    _, obj = self._parser.nextobject()
    return obj