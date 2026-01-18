from __future__ import annotations
from .mixins import UpdateDictMixin
def _del_value(self, key):
    """Used internally by the accessor properties."""
    if key in self:
        del self[key]