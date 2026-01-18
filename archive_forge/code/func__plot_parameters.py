import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _plot_parameters(self):
    """Adds a plot of temperature and Ediff to the plot."""
    steps, Ts, ediffs = ([], [], [])
    for step, line in enumerate(self._data):
        steps.extend([step + 0.5, step + 1.5])
        Ts.extend([line[2]] * 2)
        ediffs.extend([line[3]] * 2)
    self._ax.tempax.plot(steps, Ts)
    self._ax.ediffax.plot(steps, ediffs)
    for ax in [self._ax.tempax, self._ax.ediffax]:
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        ax.set_ylim((ylim[0] - 0.1 * yrange, ylim[1] + 0.1 * yrange))