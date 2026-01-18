import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _calculateconstants(self):
    """(Re)calculate some constants when pfactor, ttime or temperature have been changed."""
    n = self._getnatoms()
    if self.ttime is None:
        self.tfact = 0.0
    else:
        self.tfact = 2.0 / (3 * n * self.temperature * self.ttime * self.ttime)
    if self.pfactor_given is None:
        self.pfact = 0.0
    else:
        self.pfact = 1.0 / (self.pfactor_given * linalg.det(self._getbox()))
    self.desiredEkin = 1.5 * (n - 1) * self.temperature