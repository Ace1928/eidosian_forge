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
class GruneisenPlotter:
    """Class to plot Gruneisenparameter Object."""

    def __init__(self, gruneisen: GruneisenParameter) -> None:
        """Class to plot information from Gruneisenparameter Object.

        Args:
            gruneisen: GruneisenParameter Object.
        """
        self._gruneisen = gruneisen

    def get_plot(self, marker: str='o', markersize: float | None=None, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> Axes:
        """Will produce a plot.

        Args:
            marker: marker for the depiction
            markersize: size of the marker
            units: unit for the plots, accepted units: thz, ev, mev, ha, cm-1, cm^-1.

        Returns:
            plt.Axes: matplotlib axes object
        """
        u = freq_units(units)
        xs = self._gruneisen.frequencies.flatten() * u.factor
        ys = self._gruneisen.gruneisen.flatten()
        ax = pretty_plot(12, 8)
        ax.set_xlabel(f'$\\mathrm{{Frequency\\ ({u.label})}}$')
        ax.set_ylabel('$\\mathrm{Grüneisen\\ parameter}$')
        n = len(ys) - 1
        for idx, (xi, yi) in enumerate(zip(xs, ys)):
            color = (1.0 / n * idx, 0, 1.0 / n * (n - idx))
            ax.plot(xi, yi, marker, color=color, markersize=markersize)
        plt.tight_layout()
        return ax

    def show(self, units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Will show the plot.

        Args:
            units: units for the plot, accepted units: thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(units=units)
        plt.show()

    def save_plot(self, filename: str | PathLike, img_format: str='pdf', units: Literal['thz', 'ev', 'mev', 'ha', 'cm-1', 'cm^-1']='thz') -> None:
        """Will save the plot to a file.

        Args:
            filename: name of the filename
            img_format: format of the saved plot
            units: accepted units: thz, ev, mev, ha, cm-1, cm^-1.
        """
        self.get_plot(units=units)
        plt.savefig(filename, format=img_format)
        plt.close()