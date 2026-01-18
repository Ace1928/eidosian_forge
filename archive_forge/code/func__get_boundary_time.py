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
def _get_boundary_time(self, which: Any) -> Optional[bytes]:
    return _get_asn1_time(which(self._x509))