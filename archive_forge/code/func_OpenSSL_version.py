import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def OpenSSL_version(type):
    """
    Return a string describing the version of OpenSSL in use.

    :param type: One of the :const:`OPENSSL_` constants defined in this module.
    """
    return _ffi.string(_lib.OpenSSL_version(type))