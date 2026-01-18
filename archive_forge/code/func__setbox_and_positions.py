import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _setbox_and_positions(self, h, q):
    """Set the computational box and the positions."""
    self.atoms.set_cell(h)
    r = np.dot(q + 0.5, h)
    self.atoms.set_positions(r)