from __future__ import annotations
import binascii
import re
import sys
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.x509.oid import NameOID, ObjectIdentifier
def _parse_na(self) -> NameAttribute:
    try:
        oid_value = self._read_re(self._OID_RE)
    except ValueError:
        name = self._read_re(self._DESCR_RE)
        oid = self._attr_name_overrides.get(name, _NAME_TO_NAMEOID.get(name))
        if oid is None:
            raise ValueError
    else:
        oid = ObjectIdentifier(oid_value)
    self._read_char('=')
    if self._peek() == '#':
        value = self._read_re(self._HEXSTRING_RE)
        value = binascii.unhexlify(value[1:]).decode()
    else:
        raw_value = self._read_re(self._STRING_RE)
        value = _unescape_dn_value(raw_value)
    return NameAttribute(oid, value)