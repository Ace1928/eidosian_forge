import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
@staticmethod
def _cert_stack_to_list(cert_stack):
    """
        Internal helper to convert a STACK_OF(X509) to a list of X509
        instances.
        """
    result = []
    for i in range(_lib.sk_X509_num(cert_stack)):
        cert = _lib.sk_X509_value(cert_stack, i)
        _openssl_assert(cert != _ffi.NULL)
        res = _lib.X509_up_ref(cert)
        _openssl_assert(res >= 1)
        pycert = X509._from_raw_x509_ptr(cert)
        result.append(pycert)
    return result