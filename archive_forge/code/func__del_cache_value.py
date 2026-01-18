from __future__ import annotations
from .mixins import ImmutableDictMixin
from .mixins import UpdateDictMixin
from .. import http
def _del_cache_value(self, key):
    """Used internally by the accessor properties."""
    if key in self:
        del self[key]