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
class ThermoPlotter:
    """Plotter for thermodynamic properties obtained from phonon DOS.
    If the structure corresponding to the DOS, it will be used to extract the formula unit and provide
    the plots in units of mol instead of mole-cell.
    """

    def __init__(self, dos: PhononDos, structure: Structure=None) -> None:
        """
        Args:
            dos: A PhononDos object.
            structure: A Structure object corresponding to the structure used for the calculation.
        """
        self.dos = dos
        self.structure = structure

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

    @add_fig_kwargs
    def plot_cv(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
        """Plots the constant volume specific heat C_v in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)
        ylabel = '$C_v$ (J/K/mol)' if self.structure else '$C_v$ (J/K/mol-c)'
        return self._plot_thermo(self.dos.cv, temperatures, ylabel=ylabel, ylim=ylim, **kwargs)

    @add_fig_kwargs
    def plot_entropy(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
        """Plots the vibrational entrpy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)
        ylabel = '$S$ (J/K/mol)' if self.structure else '$S$ (J/K/mol-c)'
        return self._plot_thermo(self.dos.entropy, temperatures, ylabel=ylabel, ylim=ylim, **kwargs)

    @add_fig_kwargs
    def plot_internal_energy(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
        """Plots the vibrational internal energy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)
        ylabel = '$\\Delta E$ (kJ/mol)' if self.structure else '$\\Delta E$ (kJ/mol-c)'
        return self._plot_thermo(self.dos.internal_energy, temperatures, ylabel=ylabel, ylim=ylim, factor=0.001, **kwargs)

    @add_fig_kwargs
    def plot_helmholtz_free_energy(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
        """Plots the vibrational contribution to the Helmoltz free energy in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)
        ylabel = '$\\Delta F$ (kJ/mol)' if self.structure else '$\\Delta F$ (kJ/mol-c)'
        return self._plot_thermo(self.dos.helmholtz_free_energy, temperatures, ylabel=ylabel, ylim=ylim, factor=0.001, **kwargs)

    @add_fig_kwargs
    def plot_thermodynamic_properties(self, tmin: float, tmax: float, ntemp: int, ylim: float | None=None, **kwargs) -> Figure:
        """Plots all the thermodynamic properties in a temperature range.

        Args:
            tmin: minimum temperature
            tmax: maximum temperature
            ntemp: number of steps
            ylim: tuple specifying the y-axis limits.
            kwargs: kwargs passed to the matplotlib function 'plot'.

        Returns:
            plt.figure: matplotlib figure
        """
        temperatures = np.linspace(tmin, tmax, ntemp)
        mol = '' if self.structure else '-c'
        fig = self._plot_thermo(self.dos.cv, temperatures, ylabel='Thermodynamic properties', ylim=ylim, label=f'$C_v$ (J/K/mol{mol})', **kwargs)
        self._plot_thermo(self.dos.entropy, temperatures, ylim=ylim, ax=fig.axes[0], label=f'$S$ (J/K/mol{mol})', **kwargs)
        self._plot_thermo(self.dos.internal_energy, temperatures, ylim=ylim, ax=fig.axes[0], factor=0.001, label=f'$\\Delta E$ (kJ/mol{mol})', **kwargs)
        self._plot_thermo(self.dos.helmholtz_free_energy, temperatures, ylim=ylim, ax=fig.axes[0], factor=0.001, label=f'$\\Delta F$ (kJ/mol{mol})', **kwargs)
        fig.axes[0].legend(loc='best')
        return fig