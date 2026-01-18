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
def _getobj_objstm(self, stream: PDFStream, index: int, objid: int) -> object:
    if stream.objid in self._parsed_objs:
        objs, n = self._parsed_objs[stream.objid]
    else:
        objs, n = self._get_objects(stream)
        if self.caching:
            assert stream.objid is not None
            self._parsed_objs[stream.objid] = (objs, n)
    i = n * 2 + index
    try:
        obj = objs[i]
    except IndexError:
        raise PDFSyntaxError('index too big: %r' % index)
    return obj