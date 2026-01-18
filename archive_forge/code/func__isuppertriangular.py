import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
@staticmethod
def _isuppertriangular(m) -> bool:
    """Check that a matrix is on upper triangular form."""
    return m[1, 0] == m[2, 0] == m[2, 1] == 0.0