import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _makeuppertriangular(self, sixvector):
    """Make an upper triangular matrix from a 6-vector."""
    return np.array(((sixvector[0], sixvector[5], sixvector[4]), (0, sixvector[1], sixvector[3]), (0, 0, sixvector[2])))