from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _equal_elements_in_array(self, arr):
    s = np.sort(arr)
    return np.any(s[1:] == s[:-1])