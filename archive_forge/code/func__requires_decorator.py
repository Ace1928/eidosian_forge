import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _requires_decorator(func):
    if not flag:

        @wraps(func)
        def explode(*args, **kwargs):
            raise NotImplementedError(error)
        return explode
    else:
        return func