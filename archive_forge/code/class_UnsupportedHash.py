from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class UnsupportedHash(ValueError):
    """The requested hash algorithm is not supported.

    This exception will be thrown if a hash algorithm is requested that is
    not supported by hashlib.

    """