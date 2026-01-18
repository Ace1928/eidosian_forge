import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _startup(self):
    """Initiates a run, and determines if running from previous data or
        a fresh run."""
    status = np.array(-1.0)
    exists = self._read_minima()
    if world.rank == 0:
        if not exists:
            status = np.array(0.0)
        elif not os.path.exists(self._logfile):
            status = np.array(1.0)
        else:
            status = np.array(2.0)
    world.barrier()
    world.broadcast(status, 0)
    if status == 2.0:
        self._resume()
    else:
        self._counter = 0
        self._log('init')
        self._log('msg', 'Performing initial optimization.')
        if status == 1.0:
            self._log('msg', 'Using existing minima file with %i prior minima: %s' % (len(self._minima), self._minima_traj))
        self._optimize()
        self._check_results()
        self._counter += 1