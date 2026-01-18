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
def _untested_error(where: str) -> NoReturn:
    """
    An OpenSSL API failed somehow.  Additionally, the failure which was
    encountered isn't one that's exercised by the test suite so future behavior
    of pyOpenSSL is now somewhat less predictable.
    """
    raise RuntimeError('Unknown %s failure' % (where,))