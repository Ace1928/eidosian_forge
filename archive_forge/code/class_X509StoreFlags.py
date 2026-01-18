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
class X509StoreFlags:
    """
    Flags for X509 verification, used to change the behavior of
    :class:`X509Store`.

    See `OpenSSL Verification Flags`_ for details.

    .. _OpenSSL Verification Flags:
        https://www.openssl.org/docs/manmaster/man3/X509_VERIFY_PARAM_set_flags.html
    """
    CRL_CHECK: int = _lib.X509_V_FLAG_CRL_CHECK
    CRL_CHECK_ALL: int = _lib.X509_V_FLAG_CRL_CHECK_ALL
    IGNORE_CRITICAL: int = _lib.X509_V_FLAG_IGNORE_CRITICAL
    X509_STRICT: int = _lib.X509_V_FLAG_X509_STRICT
    ALLOW_PROXY_CERTS: int = _lib.X509_V_FLAG_ALLOW_PROXY_CERTS
    POLICY_CHECK: int = _lib.X509_V_FLAG_POLICY_CHECK
    EXPLICIT_POLICY: int = _lib.X509_V_FLAG_EXPLICIT_POLICY
    INHIBIT_MAP: int = _lib.X509_V_FLAG_INHIBIT_MAP
    CHECK_SS_SIGNATURE: int = _lib.X509_V_FLAG_CHECK_SS_SIGNATURE
    PARTIAL_CHAIN: int = _lib.X509_V_FLAG_PARTIAL_CHAIN