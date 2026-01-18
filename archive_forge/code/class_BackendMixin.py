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
class BackendMixin(PasswordHash):
    """
    PasswordHash mixin which provides generic framework for supporting multiple backends
    within the class.

    Public API
    ----------

    .. attribute:: backends

        This attribute should be a tuple containing the names of the backends
        which are supported. Two common names are ``"os_crypt"`` (if backend
        uses :mod:`crypt`), and ``"builtin"`` (if the backend is a pure-python
        fallback).

    .. automethod:: get_backend
    .. automethod:: set_backend
    .. automethod:: has_backend

    .. warning::

        :meth:`set_backend` is intended to be called during application startup --
        it affects global state, and switching backends is not guaranteed threadsafe.

    Private API (Subclass Hooks)
    ----------------------------
    Subclasses should set the :attr:`!backends` attribute to a tuple of the backends
    they wish to support.  They should also define one method:

    .. classmethod:: _load_backend_{name}(dryrun=False)

        One copy of this method should be defined for each :samp:`name` within :attr:`!backends`.

        It will be called in order to load the backend, and should take care of whatever
        is needed to enable the backend.  This may include importing modules, running tests,
        issuing warnings, etc.

        :param name:
            [Optional] name of backend.

        :param dryrun:
            [Optional] True/False if currently performing a "dry run".

            if True, the method should perform all setup actions *except*
            switching the class over to the new backend.

        :raises passlib.exc.PasslibSecurityError:
            if the backend is available, but cannot be loaded due to a security issue.

        :returns:
            False if backend not available, True if backend loaded.

        .. warning::

            Due to the way passlib's internals are arranged,
            backends should generally store stateful data at the class level
            (not the module level), and be prepared to be called on subclasses
            which may be set to a different backend from their parent.

            (Idempotent module-level data such as lazy imports are fine).

    .. automethod:: _finalize_backend

    .. versionadded:: 1.7
    """
    backends = None
    __backend = None
    _no_backend_suggestion = None
    _pending_backend = None
    _pending_dry_run = False

    @classmethod
    def get_backend(cls):
        """
        Return name of currently active backend.
        if no backend has been loaded, loads and returns name of default backend.

        :raises passlib.exc.MissingBackendError:
            if no backends are available.

        :returns:
            name of active backend
        """
        if not cls.__backend:
            cls.set_backend()
            assert cls.__backend, 'set_backend() failed to load a default backend'
        return cls.__backend

    @classmethod
    def has_backend(cls, name='any'):
        """
        Check if support is currently available for specified backend.

        :arg name:
            name of backend to check for.
            can be any string accepted by :meth:`set_backend`.

        :raises ValueError:
            if backend name is unknown

        :returns:
            * ``True`` if backend is available.
            * ``False`` if it's available / can't be loaded.
            * ``None`` if it's present, but won't load due to a security issue.
        """
        try:
            cls.set_backend(name, dryrun=True)
            return True
        except (exc.MissingBackendError, exc.PasslibSecurityError):
            return False

    @classmethod
    def set_backend(cls, name='any', dryrun=False):
        """
        Load specified backend.

        :arg name:
            name of backend to load, can be any of the following:

            * ``"any"`` -- use current backend if one is loaded,
              otherwise load the first available backend.

            * ``"default"`` -- use the first available backend.

            * any string in :attr:`backends`, loads specified backend.

        :param dryrun:
            If True, this perform all setup actions *except* switching over to the new backend.
            (this flag is used to implement :meth:`has_backend`).

            .. versionadded:: 1.7

        :raises ValueError:
            If backend name is unknown.

        :raises passlib.exc.MissingBackendError:
            If specific backend is missing;
            or in the case of ``"any"`` / ``"default"``, if *no* backends are available.

        :raises passlib.exc.PasslibSecurityError:

            If ``"any"`` or ``"default"`` was specified,
            but the only backend available has a PasslibSecurityError.
        """
        if name == 'any' and cls.__backend or (name and name == cls.__backend):
            return cls.__backend
        owner = cls._get_backend_owner()
        if owner is not cls:
            return owner.set_backend(name, dryrun=dryrun)
        if name == 'any' or name == 'default':
            default_error = None
            for name in cls.backends:
                try:
                    return cls.set_backend(name, dryrun=dryrun)
                except exc.MissingBackendError:
                    continue
                except exc.PasslibSecurityError as err:
                    if default_error is None:
                        default_error = err
                    continue
            if default_error is None:
                msg = '%s: no backends available' % cls.name
                if cls._no_backend_suggestion:
                    msg += cls._no_backend_suggestion
                default_error = exc.MissingBackendError(msg)
            raise default_error
        if name not in cls.backends:
            raise exc.UnknownBackendError(cls, name)
        with _backend_lock:
            orig = (cls._pending_backend, cls._pending_dry_run)
            try:
                cls._pending_backend = name
                cls._pending_dry_run = dryrun
                cls._set_backend(name, dryrun)
            finally:
                cls._pending_backend, cls._pending_dry_run = orig
            if not dryrun:
                cls.__backend = name
            return name

    @classmethod
    def _get_backend_owner(cls):
        """
        return class that set_backend() should actually be modifying.
        for SubclassBackendMixin, this may not always be the class that was invoked.
        """
        return cls

    @classmethod
    def _set_backend(cls, name, dryrun):
        """
        Internal method invoked by :meth:`set_backend`.
        handles actual loading of specified backend.

        global _backend_lock will be held for duration of this method,
        and _pending_dry_run & _pending_backend will also be set.

        should return True / False.
        """
        loader = cls._get_backend_loader(name)
        kwds = {}
        if accepts_keyword(loader, 'name'):
            kwds['name'] = name
        if accepts_keyword(loader, 'dryrun'):
            kwds['dryrun'] = dryrun
        ok = loader(**kwds)
        if ok is False:
            raise exc.MissingBackendError('%s: backend not available: %s' % (cls.name, name))
        elif ok is not True:
            raise AssertionError('backend loaders must return True or False: %r' % (ok,))

    @classmethod
    def _get_backend_loader(cls, name):
        """
        Hook called to get the specified backend's loader.
        Should return callable which optionally takes ``"name"`` and/or
        ``"dryrun"`` keywords.

        Callable should return True if backend initialized successfully.

        If backend can't be loaded, callable should return False
        OR raise MissingBackendError directly.
        """
        raise NotImplementedError('implement in subclass')

    @classmethod
    def _stub_requires_backend(cls):
        """
        helper for subclasses to create stub methods which auto-load backend.
        """
        if cls.__backend:
            raise AssertionError('%s: _finalize_backend(%r) failed to replace lazy loader' % (cls.name, cls.__backend))
        cls.set_backend()
        if not cls.__backend:
            raise AssertionError('%s: set_backend() failed to load a default backend' % cls.name)