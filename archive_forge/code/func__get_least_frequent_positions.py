from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _get_least_frequent_positions(self, atoms):
    """Get the positions of the least frequent element in atoms."""
    pos = atoms.get_positions(wrap=True)
    return pos[atoms.numbers == self.least_freq_element]