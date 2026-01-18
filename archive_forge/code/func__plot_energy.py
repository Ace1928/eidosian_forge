import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _plot_energy(self, step, line):
    """Plots energy and annotation for acceptance."""
    energy, status = (line[0], line[1])
    if np.isnan(energy):
        return
    self._ax.plot([step, step + 0.5], [energy] * 2, '-', color='k', linewidth=2.0)
    if status == 'accepted':
        self._ax.text(step + 0.51, energy, '$\\checkmark$')
    elif status == 'rejected':
        self._ax.text(step + 0.51, energy, '$\\Uparrow$', color='red')
    elif status == 'previously found minimum':
        self._ax.text(step + 0.51, energy, '$\\hookleftarrow$', color='red', va='center')
    elif status == 'previous minimum':
        self._ax.text(step + 0.51, energy, '$\\leftarrow$', color='red', va='center')