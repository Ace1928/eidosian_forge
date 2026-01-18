from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _get_element_count(self, struct):
    """Count the number of elements in each of the structures."""
    return Counter(struct.numbers)