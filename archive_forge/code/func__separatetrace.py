import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _separatetrace(self, mat):
    """return two matrices, one proportional to the identity
        the other traceless, which sum to the given matrix
        """
    tracePart = (mat[0][0] + mat[1][1] + mat[2][2]) / 3.0 * np.identity(3)
    return (tracePart, mat - tracePart)