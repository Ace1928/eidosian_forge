from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
def _calc_checksum_backend(self, secret):
    """
        stub for _calc_checksum_backend() --
        should load backend if one hasn't been loaded;
        if one has been loaded, this method should have been monkeypatched by _finalize_backend().
        """
    self._stub_requires_backend()
    return self._calc_checksum_backend(secret)