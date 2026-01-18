from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
class _PatchManager(object):
    """helper to manage monkeypatches and run sanity checks"""

    def __init__(self, log=None):
        self.log = log or logging.getLogger(__name__ + '._PatchManager')
        self._state = {}

    def isactive(self):
        return bool(self._state)
    __bool__ = __nonzero__ = isactive

    def _import_path(self, path):
        """retrieve obj and final attribute name from resource path"""
        name, attr = path.split(':')
        obj = __import__(name, fromlist=[attr], level=0)
        while '.' in attr:
            head, attr = attr.split('.', 1)
            obj = getattr(obj, head)
        return (obj, attr)

    @staticmethod
    def _is_same_value(left, right):
        """check if two values are the same (stripping method wrappers, etc)"""
        return get_method_function(left) == get_method_function(right)

    def _get_path(self, key, default=_UNSET):
        obj, attr = self._import_path(key)
        return getattr(obj, attr, default)

    def get(self, path, default=None):
        """return current value for path"""
        return self._get_path(path, default)

    def getorig(self, path, default=None):
        """return original (unpatched) value for path"""
        try:
            value, _ = self._state[path]
        except KeyError:
            value = self._get_path(path)
        return default if value is _UNSET else value

    def check_all(self, strict=False):
        """run sanity check on all keys, issue warning if out of sync"""
        same = self._is_same_value
        for path, (orig, expected) in iteritems(self._state):
            if same(self._get_path(path), expected):
                continue
            msg = 'another library has patched resource: %r' % path
            if strict:
                raise RuntimeError(msg)
            else:
                warn(msg, PasslibRuntimeWarning)

    def _set_path(self, path, value):
        obj, attr = self._import_path(path)
        if value is _UNSET:
            if hasattr(obj, attr):
                delattr(obj, attr)
        else:
            setattr(obj, attr, value)

    def patch(self, path, value, wrap=False):
        """monkeypatch object+attr at <path> to have <value>, stores original"""
        assert value != _UNSET
        current = self._get_path(path)
        try:
            orig, expected = self._state[path]
        except KeyError:
            self.log.debug('patching resource: %r', path)
            orig = current
        else:
            self.log.debug('modifying resource: %r', path)
            if not self._is_same_value(current, expected):
                warn('overridding resource another library has patched: %r' % path, PasslibRuntimeWarning)
        if wrap:
            assert callable(value)
            wrapped = orig
            wrapped_by = value

            def wrapper(*args, **kwds):
                return wrapped_by(wrapped, *args, **kwds)
            update_wrapper(wrapper, value)
            value = wrapper
        if callable(value):
            get_method_function(value)._patched_original_value = orig
        self._set_path(path, value)
        self._state[path] = (orig, value)

    @classmethod
    def peek_unpatched_func(cls, value):
        return value._patched_original_value

    def monkeypatch(self, parent, name=None, enable=True, wrap=False):
        """function decorator which patches function of same name in <parent>"""

        def builder(func):
            if enable:
                sep = '.' if ':' in parent else ':'
                path = parent + sep + (name or func.__name__)
                self.patch(path, func, wrap=wrap)
            return func
        if callable(name):
            func = name
            name = None
            builder(func)
            return None
        return builder

    def unpatch(self, path, unpatch_conflicts=True):
        try:
            orig, expected = self._state[path]
        except KeyError:
            return
        current = self._get_path(path)
        self.log.debug('unpatching resource: %r', path)
        if not self._is_same_value(current, expected):
            if unpatch_conflicts:
                warn('reverting resource another library has patched: %r' % path, PasslibRuntimeWarning)
            else:
                warn('not reverting resource another library has patched: %r' % path, PasslibRuntimeWarning)
                del self._state[path]
                return
        self._set_path(path, orig)
        del self._state[path]

    def unpatch_all(self, **kwds):
        for key in list(self._state):
            self.unpatch(key, **kwds)