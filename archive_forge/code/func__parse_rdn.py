from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _parse_rdn(self) -> RelativeDistinguishedName:
    nas = [self._parse_na()]
    while self._peek() == '+':
        self._read_char('+')
        nas.append(self._parse_na())
    return RelativeDistinguishedName(nas)