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
@staticmethod
def _format_page_label(value: int, style: Any) -> str:
    """Format page label value in a specific style"""
    if style is None:
        label = ''
    elif style is LIT('D'):
        label = str(value)
    elif style is LIT('R'):
        label = format_int_roman(value).upper()
    elif style is LIT('r'):
        label = format_int_roman(value)
    elif style is LIT('A'):
        label = format_int_alpha(value).upper()
    elif style is LIT('a'):
        label = format_int_alpha(value)
    else:
        log.warning('Unknown page label style: %r', style)
        label = ''
    return label