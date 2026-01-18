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
def _set_wrapped(self, handler):
    if 'ident' in handler.setting_kwds and self.orig_prefix:
        warn("PrefixWrapper: 'orig_prefix' option may not work correctly for handlers which have multiple identifiers: %r" % (handler.name,), exc.PasslibRuntimeWarning)
    self._wrapped_handler = handler