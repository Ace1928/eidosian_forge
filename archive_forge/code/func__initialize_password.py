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
def _initialize_password(self, password: str='') -> None:
    assert self.encryption is not None
    docid, param = self.encryption
    if literal_name(param.get('Filter')) != 'Standard':
        raise PDFEncryptionError('Unknown filter: param=%r' % param)
    v = int_value(param.get('V', 0))
    factory = self.security_handler_registry.get(v)
    if factory is None:
        raise PDFEncryptionError('Unknown algorithm: param=%r' % param)
    handler = factory(docid, param, password)
    self.decipher = handler.decrypt
    self.is_printable = handler.is_printable()
    self.is_modifiable = handler.is_modifiable()
    self.is_extractable = handler.is_extractable()
    assert self._parser is not None
    self._parser.fallback = False
    return