from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _set_least_frequent_element(self, atoms):
    """Save the atomic number of the least frequent element."""
    elem1 = self._get_element_count(atoms)
    self.least_freq_element = elem1.most_common()[-1][0]