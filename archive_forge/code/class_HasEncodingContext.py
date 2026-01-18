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
class HasEncodingContext(GenericHandler):
    """helper for classes which require knowledge of the encoding used"""
    context_kwds = ('encoding',)
    default_encoding = 'utf-8'

    def __init__(self, encoding=None, **kwds):
        super(HasEncodingContext, self).__init__(**kwds)
        self.encoding = encoding or self.default_encoding