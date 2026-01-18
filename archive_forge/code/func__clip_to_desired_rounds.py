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
@classmethod
def _clip_to_desired_rounds(cls, rounds):
    """
        helper for :meth:`_generate_rounds` --
        clips rounds value to desired min/max set by class (if any)
        """
    mnd = cls.min_desired_rounds or 0
    if rounds < mnd:
        return mnd
    mxd = cls.max_desired_rounds
    if mxd and rounds > mxd:
        return mxd
    return rounds