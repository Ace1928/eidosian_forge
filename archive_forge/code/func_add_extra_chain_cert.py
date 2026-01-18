import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def add_extra_chain_cert(self, certobj):
    """
        Add certificate to chain

        :param certobj: The X509 certificate object to add to the chain
        :return: None
        """
    if not isinstance(certobj, X509):
        raise TypeError('certobj must be an X509 instance')
    copy = _lib.X509_dup(certobj._x509)
    add_result = _lib.SSL_CTX_add_extra_chain_cert(self._context, copy)
    if not add_result:
        _lib.X509_free(copy)
        _raise_current_error()