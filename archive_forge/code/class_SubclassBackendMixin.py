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
class SubclassBackendMixin(BackendMixin):
    """
    variant of BackendMixin which allows backends to be implemented
    as separate mixin classes, and dynamically switches them out.

    backend classes should implement a _load_backend() classmethod,
    which will be invoked with an optional 'dryrun' keyword,
    and should return True or False.

    _load_backend() will be invoked with ``cls`` equal to the mixin,
    *not* the overall class.

    .. versionadded:: 1.7
    """
    _backend_mixin_target = False
    _backend_mixin_map = None

    @classmethod
    def _get_backend_owner(cls):
        """
        return base class that we're actually switching backends on
        (needed in since backends frequently modify class attrs,
        and .set_backend may be called from a subclass).
        """
        if not cls._backend_mixin_target:
            raise AssertionError('_backend_mixin_target not set')
        for base in cls.__mro__:
            if base.__dict__.get('_backend_mixin_target'):
                return base
        raise AssertionError("expected to find class w/ '_backend_mixin_target' set")

    @classmethod
    def _set_backend(cls, name, dryrun):
        super(SubclassBackendMixin, cls)._set_backend(name, dryrun)
        assert cls is cls._get_backend_owner(), '_finalize_backend() not invoked on owner'
        mixin_map = cls._backend_mixin_map
        assert mixin_map, '_backend_mixin_map not specified'
        mixin_cls = mixin_map[name]
        assert issubclass(mixin_cls, SubclassBackendMixin), 'invalid mixin class'
        update_mixin_classes(cls, add=mixin_cls, remove=mixin_map.values(), append=True, before=SubclassBackendMixin, dryrun=dryrun)

    @classmethod
    def _get_backend_loader(cls, name):
        assert cls._backend_mixin_map, '_backend_mixin_map not specified'
        return cls._backend_mixin_map[name]._load_backend_mixin