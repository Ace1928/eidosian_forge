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
def _get_backend_loader(cls, name):
    """
        subclassed to support legacy 1.6 HasManyBackends api.
        (will be removed in passlib 2.0)
        """
    loader = getattr(cls, '_load_backend_' + name, None)
    if loader is None:

        def loader():
            return cls.__load_legacy_backend(name)
    else:
        assert not hasattr(cls, '_has_backend_' + name), "%s: can't specify both ._load_backend_%s() and ._has_backend_%s" % (cls.name, name, name)
    return loader