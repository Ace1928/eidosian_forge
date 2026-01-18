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
class ParallelismMixin(GenericHandler):
    """
    mixin which provides common behavior for 'parallelism' setting
    """
    parallelism = 1

    @classmethod
    def using(cls, parallelism=None, **kwds):
        subcls = super(ParallelismMixin, cls).using(**kwds)
        if parallelism is not None:
            if isinstance(parallelism, native_string_types):
                parallelism = int(parallelism)
            subcls.parallelism = subcls._norm_parallelism(parallelism, relaxed=kwds.get('relaxed'))
        return subcls

    def __init__(self, parallelism=None, **kwds):
        super(ParallelismMixin, self).__init__(**kwds)
        if parallelism is None:
            assert validate_default_value(self, self.parallelism, self._norm_parallelism, param='parallelism')
        else:
            self.parallelism = self._norm_parallelism(parallelism)

    @classmethod
    def _norm_parallelism(cls, parallelism, relaxed=False):
        return norm_integer(cls, parallelism, min=1, param='parallelism', relaxed=relaxed)

    def _calc_needs_update(self, **kwds):
        """
        mark hash as needing update if rounds is outside desired bounds.
        """
        if self.parallelism != type(self).parallelism:
            return True
        return super(ParallelismMixin, self)._calc_needs_update(**kwds)