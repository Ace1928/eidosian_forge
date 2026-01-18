import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _synchronize(self):
    """Synchronizes eta, h and zeta on all processors in a parallel simulation.

        In a parallel simulation, eta, h and zeta are communicated
        from the master to all slaves, to prevent numerical noise from
        causing them to diverge.

        In a serial simulation, do nothing.
        """
    pass