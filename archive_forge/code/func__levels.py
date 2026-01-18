import torch
from . import _Tensor, Tensor
from .reference import _dims, _enable_layers, llist, ltuple
@property
def _levels(self):
    if self._levels_data is None:
        levels = llist(self._lhs._levels)
        for l in self._rhs._levels:
            if l not in levels:
                levels.append(l)
        self._levels_data = ltuple(levels)
    return self._levels_data