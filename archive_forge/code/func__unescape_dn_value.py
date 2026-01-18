from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _unescape_dn_value(val: str) -> str:
    if not val:
        return ''

    def sub(m):
        val = m.group(1)
        if len(val) == 1:
            return val
        return chr(int(val, 16))
    return _RFC4514NameParser._PAIR_RE.sub(sub, val)