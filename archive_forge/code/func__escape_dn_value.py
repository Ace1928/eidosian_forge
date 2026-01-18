from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _escape_dn_value(val: typing.Union[str, bytes]) -> str:
    """Escape special characters in RFC4514 Distinguished Name value."""
    if not val:
        return ''
    if isinstance(val, bytes):
        return '#' + binascii.hexlify(val).decode('utf8')
    val = val.replace('\\', '\\\\')
    val = val.replace('"', '\\"')
    val = val.replace('+', '\\+')
    val = val.replace(',', '\\,')
    val = val.replace(';', '\\;')
    val = val.replace('<', '\\<')
    val = val.replace('>', '\\>')
    val = val.replace('\x00', '\\00')
    if val[0] in ('#', ' '):
        val = '\\' + val
    if val[-1] == ' ':
        val = val[:-1] + '\\ '
    return val