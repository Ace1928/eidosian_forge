import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
class _VerifyHelper(_CallbackExceptionHelper):
    """
    Wrap a callback such that it can be used as a certificate verification
    callback.
    """

    def __init__(self, callback):
        _CallbackExceptionHelper.__init__(self)

        @wraps(callback)
        def wrapper(ok, store_ctx):
            x509 = _lib.X509_STORE_CTX_get_current_cert(store_ctx)
            _lib.X509_up_ref(x509)
            cert = X509._from_raw_x509_ptr(x509)
            error_number = _lib.X509_STORE_CTX_get_error(store_ctx)
            error_depth = _lib.X509_STORE_CTX_get_error_depth(store_ctx)
            index = _lib.SSL_get_ex_data_X509_STORE_CTX_idx()
            ssl = _lib.X509_STORE_CTX_get_ex_data(store_ctx, index)
            connection = Connection._reverse_mapping[ssl]
            try:
                result = callback(connection, cert, error_number, error_depth, ok)
            except Exception as e:
                self._problems.append(e)
                return 0
            else:
                if result:
                    _lib.X509_STORE_CTX_set_error(store_ctx, _lib.X509_V_OK)
                    return 1
                else:
                    return 0
        self.callback = _ffi.callback('int (*)(int, X509_STORE_CTX *)', wrapper)