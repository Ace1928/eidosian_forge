import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _plot_qn(self, index, line):
    """Plots a dashed vertical line for the optimization."""
    if line[1] == 'performing MD':
        return
    file = os.path.join(self._rundirectory, 'qn%05i.traj' % index)
    if os.path.getsize(file) == 0:
        return
    with io.Trajectory(file, 'r') as traj:
        energies = [traj[0].get_potential_energy(), traj[-1].get_potential_energy()]
    if index > 0:
        file = os.path.join(self._rundirectory, 'md%05i.traj' % index)
        atoms = io.read(file, index=-3)
        energies[0] = atoms.get_potential_energy()
    self._ax.plot([index + 0.25] * 2, energies, ':k')