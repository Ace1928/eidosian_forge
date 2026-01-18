from torch.ao.pruning import BaseSparsifier
from functools import wraps
import warnings
import weakref
def _make_sure_a_list(self, var):
    """Utility that extends it to the same length as the .groups, ensuring it is a list"""
    n = len(self.sparsifier.groups)
    if not isinstance(var, (list, tuple)):
        return [var] * n
    else:
        if len(var) != n:
            raise ValueError(f'Expected variable of length {n}, but got {len(var)}')
        return list(var)