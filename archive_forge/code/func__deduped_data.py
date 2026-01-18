import numpy as np
from ._base import _spbase, _ufuncs_with_fixed_point_at_zero
from ._sputils import isscalarlike, validateaxis
def _deduped_data(self):
    if hasattr(self, 'sum_duplicates'):
        self.sum_duplicates()
    return self.data