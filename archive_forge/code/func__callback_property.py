from __future__ import annotations
from .. import http
def _callback_property(name):

    def fget(self):
        return getattr(self, name)

    def fset(self, value):
        setattr(self, name, value)
        if self.on_update is not None:
            self.on_update(self)
    return property(fget, fset)