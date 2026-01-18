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
def _set_asn1_time(boundary: Any, when: bytes) -> None:
    """
    The the time value of an ASN1 time object.

    @param boundary: An ASN1_TIME pointer (or an object safely
        castable to that type) which will have its value set.
    @param when: A string representation of the desired time value.

    @raise TypeError: If C{when} is not a L{bytes} string.
    @raise ValueError: If C{when} does not represent a time in the required
        format.
    @raise RuntimeError: If the time value cannot be set for some other
        (unspecified) reason.
    """
    if not isinstance(when, bytes):
        raise TypeError('when must be a byte string')
    _openssl_assert(boundary != _ffi.NULL)
    set_result = _lib.ASN1_TIME_set_string(boundary, when)
    if set_result == 0:
        raise ValueError('Invalid string')