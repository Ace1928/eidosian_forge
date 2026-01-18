from . import _gi
from ._constants import \
def _get_minimum(self):
    return self._min_value_lookup.get(self.type, None)