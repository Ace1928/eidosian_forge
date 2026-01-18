import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _makecanvas(self):
    from matplotlib import pyplot
    from matplotlib.ticker import ScalarFormatter
    fig = pyplot.figure(figsize=(6.0, 8.0))
    lm, rm, bm, tm = (0.22, 0.02, 0.05, 0.04)
    vg1 = 0.01
    vg2 = 0.03
    ratio = 2.0
    figwidth = 1.0 - lm - rm
    totalfigheight = 1.0 - bm - tm - vg1 - 2.0 * vg2
    parfigheight = totalfigheight / (2.0 * ratio + 2)
    epotheight = ratio * parfigheight
    ax1 = fig.add_axes((lm, bm, figwidth, epotheight))
    ax2 = fig.add_axes((lm, bm + epotheight + vg1, figwidth, epotheight))
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ediffax = fig.add_axes((lm, bm + 2.0 * epotheight + vg1 + vg2, figwidth, parfigheight))
    tempax = fig.add_axes((lm, bm + 2 * epotheight + vg1 + 2 * vg2 + parfigheight, figwidth, parfigheight))
    for ax in [ax2, tempax, ediffax]:
        ax.set_xticklabels([])
    ax1.set_xlabel('step')
    tempax.set_ylabel('$T$, K')
    ediffax.set_ylabel('$E_\\mathrm{diff}$, eV')
    for ax in [ax1, ax2]:
        ax.set_ylabel('$E_\\mathrm{pot}$, eV')
    ax = CombinedAxis(ax1, ax2, tempax, ediffax)
    self._set_zoomed_range(ax)
    ax1.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    return (fig, ax)