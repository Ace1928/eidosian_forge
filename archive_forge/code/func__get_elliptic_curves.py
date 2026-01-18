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
@classmethod
def _get_elliptic_curves(cls, lib: Any) -> Set['_EllipticCurve']:
    """
        Get, cache, and return the curves supported by OpenSSL.

        :param lib: The OpenSSL library binding object.

        :return: A :py:type:`set` of ``cls`` instances giving the names of the
            elliptic curves the underlying library supports.
        """
    if cls._curves is None:
        cls._curves = cls._load_elliptic_curves(lib)
    return cls._curves