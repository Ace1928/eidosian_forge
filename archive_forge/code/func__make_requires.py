import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def _make_requires(flag, error):
    """
    Builds a decorator that ensures that functions that rely on OpenSSL
    functions that are not present in this build raise NotImplementedError,
    rather than AttributeError coming out of cryptography.

    :param flag: A cryptography flag that guards the functions, e.g.
        ``Cryptography_HAS_NEXTPROTONEG``.
    :param error: The string to be used in the exception if the flag is false.
    """

    def _requires_decorator(func):
        if not flag:

            @wraps(func)
            def explode(*args, **kwargs):
                raise NotImplementedError(error)
            return explode
        else:
            return func
    return _requires_decorator