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
@property
def _stub_checksum(self):
    """
        placeholder used by default .genconfig() so it can avoid expense of calculating digest.
        """
    if self.checksum_size:
        if self._checksum_is_bytes:
            return b'\x00' * self.checksum_size
        if self.checksum_chars:
            return self.checksum_chars[0] * self.checksum_size
    if isinstance(self, HasRounds):
        orig = self.rounds
        self.rounds = self.min_rounds or 1
        try:
            return self._calc_checksum('')
        finally:
            self.rounds = orig
    return self._calc_checksum('')