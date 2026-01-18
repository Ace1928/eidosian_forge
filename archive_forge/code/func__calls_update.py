from __future__ import annotations
from itertools import repeat
from .._internal import _missing
def _calls_update(name):

    def oncall(self, *args, **kw):
        rv = getattr(super(UpdateDictMixin, self), name)(*args, **kw)
        if self.on_update is not None:
            self.on_update(self)
        return rv
    oncall.__name__ = name
    return oncall