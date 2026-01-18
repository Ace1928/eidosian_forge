from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _read_char(self, ch: str) -> None:
    if self._peek() != ch:
        raise ValueError
    self._idx += 1