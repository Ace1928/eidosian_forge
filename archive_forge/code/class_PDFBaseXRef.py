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
class PDFBaseXRef:

    def get_trailer(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_objids(self) -> Iterable[int]:
        return []

    def get_pos(self, objid: int) -> Tuple[Optional[int], int, int]:
        raise KeyError(objid)

    def load(self, parser: PDFParser) -> None:
        raise NotImplementedError