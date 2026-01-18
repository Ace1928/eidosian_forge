import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def _set_block_size(self, ndx, size):
    if ndx in self._undefined_brows:
        self._undefined_brows.remove(ndx)
        self._brow_lengths[ndx] = size
        if len(self._undefined_brows) == 0:
            self._brow_lengths = np.asarray(self._brow_lengths, dtype=np.int64)
    elif self._brow_lengths[ndx] != size:
        raise ValueError('Incompatible dimensions for block {ndx}; got {got}; expected {exp}'.format(ndx=ndx, got=size, exp=self._brow_lengths[ndx]))