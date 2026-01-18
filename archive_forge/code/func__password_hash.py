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
def _password_hash(self, password: bytes, salt: bytes, vector: Optional[bytes]=None) -> bytes:
    """
        Compute password hash depending on revision number
        """
    if self.r == 5:
        return self._r5_password(password, salt, vector)
    return self._r6_password(password, salt[0:8], vector)