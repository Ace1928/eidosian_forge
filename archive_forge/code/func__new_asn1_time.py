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
def _new_asn1_time(when: bytes) -> Any:
    """
    Behaves like _set_asn1_time but returns a new ASN1_TIME object.

    @param when: A string representation of the desired time value.

    @raise TypeError: If C{when} is not a L{bytes} string.
    @raise ValueError: If C{when} does not represent a time in the required
        format.
    @raise RuntimeError: If the time value cannot be set for some other
        (unspecified) reason.
    """
    ret = _lib.ASN1_TIME_new()
    _openssl_assert(ret != _ffi.NULL)
    ret = _ffi.gc(ret, _lib.ASN1_TIME_free)
    _set_asn1_time(ret, when)
    return ret