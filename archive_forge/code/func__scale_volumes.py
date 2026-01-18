from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _scale_volumes(self):
    """Scale the cell of s2 to have the same volume as s1."""
    cell2 = self.s2.get_cell()
    v2 = np.linalg.det(cell2)
    v1 = np.linalg.det(self.s1.get_cell())
    coordinate_scaling = (v1 / v2) ** (1.0 / 3.0)
    cell2 *= coordinate_scaling
    self.s2.set_cell(cell2, scale_atoms=True)