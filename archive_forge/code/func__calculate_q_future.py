import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _calculate_q_future(self, force):
    """Calculate future q.  Needed in Timestep and Initialization."""
    dt = self.dt
    id3 = np.identity(3)
    alpha = dt * dt * np.dot(force / self._getmasses(), self.inv_h)
    beta = dt * np.dot(self.h, np.dot(self.eta + 0.5 * self.zeta * id3, self.inv_h))
    inv_b = linalg.inv(beta + id3)
    self.q_future = np.dot(2 * self.q + np.dot(self.q_past, beta - id3) + alpha, inv_b)