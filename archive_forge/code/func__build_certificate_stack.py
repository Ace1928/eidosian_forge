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
@staticmethod
def _build_certificate_stack(certificates: Optional[Sequence[X509]]) -> None:

    def cleanup(s: Any) -> None:
        for i in range(_lib.sk_X509_num(s)):
            x = _lib.sk_X509_value(s, i)
            _lib.X509_free(x)
        _lib.sk_X509_free(s)
    if certificates is None or len(certificates) == 0:
        return _ffi.NULL
    stack = _lib.sk_X509_new_null()
    _openssl_assert(stack != _ffi.NULL)
    stack = _ffi.gc(stack, cleanup)
    for cert in certificates:
        if not isinstance(cert, X509):
            raise TypeError('One of the elements is not an X509 instance')
        _openssl_assert(_lib.X509_up_ref(cert._x509) > 0)
        if _lib.sk_X509_push(stack, cert._x509) <= 0:
            _lib.X509_free(cert._x509)
            _raise_current_error()
    return stack