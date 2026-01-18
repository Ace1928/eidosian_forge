from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _get_only_least_frequent_of(self, struct):
    """Get the atoms object with all other elements than the least frequent
        one removed. Wrap the positions to get everything in the cell."""
    pos = struct.get_positions(wrap=True)
    indices = struct.numbers == self.least_freq_element
    least_freq_struct = struct[indices]
    least_freq_struct.set_positions(pos[indices])
    return least_freq_struct