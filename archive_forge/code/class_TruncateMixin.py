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
class TruncateMixin(MinimalHandler):
    """
    PasswordHash mixin which provides a method
    that will check if secret would be truncated,
    and can be configured to throw an error.

    .. warning::

        Hashers using this mixin will generally need to override
        the default PasswordHash.truncate_error policy of "True",
        and will similarly want to override .truncate_verify_reject as well.

        TODO: This should be done explicitly, but for now this mixin sets
        these flags implicitly.
    """
    truncate_error = False
    truncate_verify_reject = False

    @classmethod
    def using(cls, truncate_error=None, **kwds):
        subcls = super(TruncateMixin, cls).using(**kwds)
        if truncate_error is not None:
            truncate_error = as_bool(truncate_error, param='truncate_error')
            if truncate_error is not None:
                subcls.truncate_error = truncate_error
        return subcls

    @classmethod
    def _check_truncate_policy(cls, secret):
        """
        make sure secret won't be truncated.
        NOTE: this should only be called for .hash(), not for .verify(),
        which should honor the .truncate_verify_reject policy.
        """
        assert cls.truncate_size is not None, 'truncate_size must be set by subclass'
        if cls.truncate_error and len(secret) > cls.truncate_size:
            raise exc.PasswordTruncateError(cls)