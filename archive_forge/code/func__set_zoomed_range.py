import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _set_zoomed_range(self, ax):
    """Try to intelligently set the range for the zoomed-in part of the
        graph."""
    energies = [line[0] for line in self._data if not np.isnan(line[0])]
    dr = max(energies) - min(energies)
    if dr == 0.0:
        dr = 1.0
    ax.set_ax1_range((min(energies) - 0.2 * dr, max(energies) + 0.2 * dr))