import warnings
import numpy as np
from ase.optimize.optimize import Dynamics
from ase.md.logger import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
def _get_com_velocity(self, velocity):
    """Return the center of mass velocity.
        Internal use only. This function can be reimplemented by Asap.
        """
    return np.dot(self.masses.ravel(), velocity) / self.masses.sum()