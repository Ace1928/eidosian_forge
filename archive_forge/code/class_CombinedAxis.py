import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
class CombinedAxis:
    """Helper class for MHPlot to plot on split y axis and adjust limits
    simultaneously."""

    def __init__(self, ax1, ax2, tempax, ediffax):
        self.ax1 = ax1
        self.ax2 = ax2
        self.tempax = tempax
        self.ediffax = ediffax
        self._ymax = -np.inf

    def set_ax1_range(self, ylim):
        self._ax1_ylim = ylim
        self.ax1.set_ylim(ylim)

    def plot(self, *args, **kwargs):
        self.ax1.plot(*args, **kwargs)
        self.ax2.plot(*args, **kwargs)
        for yvalue in args[1]:
            if yvalue > self._ymax:
                self._ymax = yvalue
        self.ax1.set_ylim(self._ax1_ylim)
        self.ax2.set_ylim((self._ax1_ylim[1], self._ymax))

    def set_xlim(self, *args):
        self.ax1.set_xlim(*args)
        self.ax2.set_xlim(*args)
        self.tempax.set_xlim(*args)
        self.ediffax.set_xlim(*args)

    def text(self, *args, **kwargs):
        y = args[1]
        if y < self._ax1_ylim[1]:
            ax = self.ax1
        else:
            ax = self.ax2
        ax.text(*args, **kwargs)