import calendar
import datetime
import functools
import typing
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
from OpenSSL._util import (
def _verify_certificate(self) -> Any:
    """
        Verifies the certificate and runs an X509_STORE_CTX containing the
        results.

        :raises X509StoreContextError: If an error occurred when validating a
          certificate in the context. Sets ``certificate`` attribute to
          indicate which certificate caused the error.
        """
    store_ctx = _lib.X509_STORE_CTX_new()
    _openssl_assert(store_ctx != _ffi.NULL)
    store_ctx = _ffi.gc(store_ctx, _lib.X509_STORE_CTX_free)
    ret = _lib.X509_STORE_CTX_init(store_ctx, self._store._store, self._cert._x509, self._chain)
    _openssl_assert(ret == 1)
    ret = _lib.X509_verify_cert(store_ctx)
    if ret <= 0:
        raise self._exception_from_context(store_ctx)
    return store_ctx