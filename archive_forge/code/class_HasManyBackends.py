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
class HasManyBackends(BackendMixin, GenericHandler):
    """
    GenericHandler mixin which provides selecting from multiple backends.

    .. todo::

        finish documenting this class's usage

    For hashes which need to select from multiple backends,
    depending on the host environment, this class
    offers a way to specify alternate :meth:`_calc_checksum` methods,
    and will dynamically chose the best one at runtime.

    .. versionchanged:: 1.7

        This class now derives from :class:`BackendMixin`, which abstracts
        out a more generic framework for supporting multiple backends.
        The public api (:meth:`!get_backend`, :meth:`!has_backend`, :meth:`!set_backend`)
        is roughly the same.

    Private API (Subclass Hooks)
    ----------------------------
    As of version 1.7, classes should implement :meth:`!_load_backend_{name}`, per
    :class:`BackendMixin`.  This hook should invoke :meth:`!_set_calc_checksum_backcend`
    to install it's backend method.

    .. deprecated:: 1.7

        The following api is deprecated, and will be removed in Passlib 2.0:

    .. attribute:: _has_backend_{name}

        private class attribute checked by :meth:`has_backend` to see if a
        specific backend is available, it should be either ``True``
        or ``False``. One of these should be provided by
        the subclass for each backend listed in :attr:`backends`.

    .. classmethod:: _calc_checksum_{name}

        private class method that should implement :meth:`_calc_checksum`
        for a given backend. it will only be called if the backend has
        been selected by :meth:`set_backend`. One of these should be provided
        by the subclass for each backend listed in :attr:`backends`.
    """

    def _calc_checksum(self, secret):
        """wrapper for backend, for common code"""
        return self._calc_checksum_backend(secret)

    def _calc_checksum_backend(self, secret):
        """
        stub for _calc_checksum_backend() --
        should load backend if one hasn't been loaded;
        if one has been loaded, this method should have been monkeypatched by _finalize_backend().
        """
        self._stub_requires_backend()
        return self._calc_checksum_backend(secret)

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

    @classmethod
    def __load_legacy_backend(cls, name):
        value = getattr(cls, '_has_backend_' + name)
        warn('%s: support for ._has_backend_%s is deprecated as of Passlib 1.7, and will be removed in Passlib 1.9/2.0, please implement ._load_backend_%s() instead' % (cls.name, name, name), DeprecationWarning)
        if value:
            func = getattr(cls, '_calc_checksum_' + name)
            cls._set_calc_checksum_backend(func)
            return True
        else:
            return False

    @classmethod
    def _set_calc_checksum_backend(cls, func):
        """
        helper used by subclasses to validate & set backend-specific
        calc checksum helper.
        """
        backend = cls._pending_backend
        assert backend, 'should only be called during set_backend()'
        if not callable(func):
            raise RuntimeError('%s: backend %r returned invalid callable: %r' % (cls.name, backend, func))
        if not cls._pending_dry_run:
            cls._calc_checksum_backend = func