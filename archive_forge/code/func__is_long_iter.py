import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _is_long_iter(self, obj, level=0):
    try:
        if isinstance(obj, self.string_types):
            return len(obj) > self.maxstring_inner
        if not hasattr(obj, '__iter__'):
            return False
        if not isinstance(obj, self.long_iter_types):
            return False
        if obj is iter(obj):
            return False
        if isinstance(obj, range):
            return False
        try:
            module = type(obj).__module__.partition('.')[0]
            if module in ('numpy', 'scipy'):
                return False
        except Exception:
            pass
        if level >= len(self.maxcollection):
            return True
        if hasattr(obj, '__len__'):
            try:
                size = len(obj)
            except Exception:
                size = None
            if size is not None and size > self.maxcollection[level]:
                return True
            return any((self._is_long_iter(item, level + 1) for item in obj))
        return any((i > self.maxcollection[level] or self._is_long_iter(item, level + 1) for i, item in enumerate(obj)))
    except Exception:
        return True