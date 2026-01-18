from __future__ import annotations
from itertools import repeat
from .._internal import _missing
class UpdateDictMixin(dict):
    """Makes dicts call `self.on_update` on modifications.

    .. versionadded:: 0.5

    :private:
    """
    on_update = None

    def setdefault(self, key, default=None):
        modified = key not in self
        rv = super().setdefault(key, default)
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv

    def pop(self, key, default=_missing):
        modified = key in self
        if default is _missing:
            rv = super().pop(key)
        else:
            rv = super().pop(key, default)
        if modified and self.on_update is not None:
            self.on_update(self)
        return rv
    __setitem__ = _calls_update('__setitem__')
    __delitem__ = _calls_update('__delitem__')
    clear = _calls_update('clear')
    popitem = _calls_update('popitem')
    update = _calls_update('update')