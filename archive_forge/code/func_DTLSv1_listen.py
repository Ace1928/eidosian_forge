import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def DTLSv1_listen(self):
    """
        Call the OpenSSL function DTLSv1_listen on this connection. See the
        OpenSSL manual for more details.

        :return: None
        """
    bio_addr = _lib.BIO_ADDR_new()
    try:
        result = _lib.DTLSv1_listen(self._ssl, bio_addr)
    finally:
        _lib.BIO_ADDR_free(bio_addr)
    if self._cookie_generate_helper is not None:
        self._cookie_generate_helper.raise_if_problem()
    if self._cookie_verify_helper is not None:
        self._cookie_verify_helper.raise_if_problem()
    if result == 0:
        raise WantReadError()
    if result < 0:
        self._raise_ssl_error(self._ssl, result)