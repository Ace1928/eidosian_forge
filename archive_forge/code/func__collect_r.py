import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def _collect_r(self, arr_ro, oshape, dtype):
    """Collect an array that is distributed."""
    if len(self.myr) == self.ndof:
        return arr_ro
    data_ro = np.zeros([self.ndof] + oshape, dtype)
    if len(arr_ro):
        data_ro[self.slize] = arr_ro
    self.comm.sum(data_ro)
    return data_ro