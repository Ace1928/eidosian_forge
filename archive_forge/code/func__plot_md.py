import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _plot_md(self, step, line):
    """Adds a curved plot of molecular dynamics trajectory."""
    if step == 0:
        return
    energies = [self._data[step - 1][0]]
    file = os.path.join(self._rundirectory, 'md%05i.traj' % step)
    with io.Trajectory(file, 'r') as traj:
        for atoms in traj:
            energies.append(atoms.get_potential_energy())
    xi = step - 1 + 0.5
    if len(energies) > 2:
        xf = xi + (step + 0.25 - xi) * len(energies) / (len(energies) - 2.0)
    else:
        xf = step
    if xf > step + 0.75:
        xf = step
    self._ax.plot(np.linspace(xi, xf, num=len(energies)), energies, '-k')