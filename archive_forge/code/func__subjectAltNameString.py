import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
def _subjectAltNameString(self) -> str:
    names = _ffi.cast('GENERAL_NAMES*', _lib.X509V3_EXT_d2i(self._extension))
    names = _ffi.gc(names, _lib.GENERAL_NAMES_free)
    parts = []
    for i in range(_lib.sk_GENERAL_NAME_num(names)):
        name = _lib.sk_GENERAL_NAME_value(names, i)
        try:
            label = self._prefixes[name.type]
        except KeyError:
            bio = _new_mem_buf()
            _lib.GENERAL_NAME_print(bio, name)
            parts.append(_bio_to_string(bio).decode('utf-8'))
        else:
            value = _ffi.buffer(name.d.ia5.data, name.d.ia5.length)[:].decode('utf-8')
            parts.append(label + ':' + value)
    return ', '.join(parts)