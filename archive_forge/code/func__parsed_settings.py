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
@classproperty
def _parsed_settings(cls):
    """
        helper for :meth:`parsehash` --
        returns list of attributes which should be extracted by parse_hash() from hasher object.

        default implementation just takes setting_kwds, and excludes _unparsed_settings
        """
    return tuple((key for key in cls.setting_kwds if key not in cls._unparsed_settings))