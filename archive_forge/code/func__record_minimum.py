import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _record_minimum(self):
    """Adds the current atoms configuration to the minima list."""
    with io.Trajectory(self._minima_traj, 'a') as traj:
        traj.write(self._atoms)
    self._read_minima()
    self._log('msg', 'Recorded minima #%i.' % (len(self._minima) - 1))