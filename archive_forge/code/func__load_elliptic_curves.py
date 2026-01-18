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
def _load_elliptic_curves(cls, lib: Any) -> Set['_EllipticCurve']:
    """
        Get the curves supported by OpenSSL.

        :param lib: The OpenSSL library binding object.

        :return: A :py:type:`set` of ``cls`` instances giving the names of the
            elliptic curves the underlying library supports.
        """
    num_curves = lib.EC_get_builtin_curves(_ffi.NULL, 0)
    builtin_curves = _ffi.new('EC_builtin_curve[]', num_curves)
    lib.EC_get_builtin_curves(builtin_curves, num_curves)
    return set((cls.from_nid(lib, c.nid) for c in builtin_curves))