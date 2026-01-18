from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Callable
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from matplotlib.collections import LineCollection
from monty.json import jsanitize
from pymatgen.electronic_structure.plotter import BSDOSPlotter, plot_brillouin_zone
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.gruneisen import GruneisenPhononBandStructureSymmLine
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
def _plot_thermo(self, func: Callable[[float, Structure | None], float], temperatures: Sequence[float], factor: float=1, ax: Axes=None, ylabel: str | None=None, label: str | None=None, ylim: float | None=None, **kwargs) -> Figure:
    """Plots a thermodynamic property for a generic function from a PhononDos instance.

        Args:
            func (Callable[[float, Structure | None], float]): Takes a temperature and structure (in that order)
                and returns a thermodynamic property (e.g., heat capacity, entropy, etc.).
            temperatures (list[float]): temperatures (in K) at which to evaluate func.
            factor: a multiplicative factor applied to the thermodynamic property calculated. Used to change
                the units. Defaults to 1.
            ax: matplotlib Axes or None if a new figure should be created.
            ylabel: label for the y axis
            label: label of the plot
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
    ax, fig = get_ax_fig(ax)
    values = []
    for temp in temperatures:
        values.append(func(temp, self.structure) * factor)
    ax.plot(temperatures, values, label=label, **kwargs)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim((np.min(temperatures), np.max(temperatures)))
    _ylim = plt.ylim()
    if _ylim[0] < 0 < _ylim[1]:
        plt.plot(plt.xlim(), [0, 0], 'k-', linewidth=1)
    ax.set_xlabel('$T$ (K)')
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig