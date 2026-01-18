from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _least_frequent_element_to_origin(self, atoms):
    """Put one of the least frequent elements at the origin."""
    least_freq_pos = self._get_least_frequent_positions(atoms)
    cell_diag = np.sum(atoms.get_cell(), axis=0)
    d = least_freq_pos[0] - 1e-06 * cell_diag
    atoms.positions -= d
    atoms.wrap(pbc=[1, 1, 1])