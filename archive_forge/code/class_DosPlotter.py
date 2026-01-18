from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
class DosPlotter:
    """Class for plotting phonon DOSs. The interface is extremely flexible given there are many
    different ways in which people want to view DOS.
    Typical usage is:
        # Initializes plotter with some optional args. Defaults are usually fine
        plotter = PhononDosPlotter().

        # Add DOS with a label
        plotter.add_dos("Total DOS", dos)

        # Alternatively, you can add a dict of DOSes. This is the typical form
        # returned by CompletePhononDos.get_element_dos().
        plotter.add_dos_dict({"dos1": dos1, "dos2": dos2})
        plotter.add_dos_dict(complete_dos.get_spd_dos())
    """

    def __init__(self, zero_at_efermi: bool=True, stack: bool=False, sigma: float | None=None) -> None:
        """
        Args:
            zero_at_efermi (bool): Whether to shift all Dos to have zero energy at the
                fermi energy. Defaults to True.
            stack (bool): Whether to plot the DOS as a stacked area graph
            sigma (float): Specify a standard deviation for Gaussian smearing
                the DOS for nicer looking plots. Defaults to None for no
                smearing.
        """
        self.zero_at_efermi = zero_at_efermi
        self.stack = stack
        self.sigma = sigma
        self._norm_val = True
        self._doses: dict[str, dict[Literal['energies', 'densities', 'efermi'], float | ArrayLike | dict[Spin, ArrayLike]]] = {}

    def add_dos(self, label: str, dos: Dos) -> None:
        """Adds a dos for plotting.

        Args:
            label: label for the DOS. Must be unique.
            dos: Dos object
        """
        if dos.norm_vol is None:
            self._norm_val = False
        energies = dos.energies - dos.efermi if self.zero_at_efermi else dos.energies
        densities = dos.get_smeared_densities(self.sigma) if self.sigma else dos.densities
        efermi = dos.efermi
        self._doses[label] = {'energies': energies, 'densities': densities, 'efermi': efermi}

    def add_dos_dict(self, dos_dict, key_sort_func=None) -> None:
        """Add a dictionary of doses, with an optional sorting function for the
        keys.

        Args:
            dos_dict: dict of {label: Dos}
            key_sort_func: function used to sort the dos_dict keys.
        """
        keys = sorted(dos_dict, key=key_sort_func) if key_sort_func else list(dos_dict)
        for label in keys:
            self.add_dos(label, dos_dict[label])

    def get_dos_dict(self):
        """Returns the added doses as a json-serializable dict. Note that if you
        have specified smearing for the DOS plot, the densities returned will
        be the smeared densities, not the original densities.

        Returns:
            dict: Dict of dos data. Generally of the form
            {label: {'energies':..., 'densities': {'up':...}, 'efermi':efermi}}
        """
        return jsanitize(self._doses)

    @typing.no_type_check
    def get_plot(self, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, invert_axes: bool=False, beta_dashed: bool=False) -> plt.Axes:
        """Get a matplotlib plot showing the DOS.

        Args:
            xlim (tuple[float, float]): The energy axis limits. Defaults to None for automatic
                determination.
            ylim (tuple[float, float]): The y-axis limits. Defaults to None for automatic determination.
            invert_axes (bool): Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            beta_dashed (bool): Plots the beta spin channel with a dashed line. Defaults to False.

        Returns:
            plt.Axes: matplotlib Axes object.
        """
        n_colors = min(9, max(3, len(self._doses)))
        colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ys = None
        all_densities = []
        all_energies = []
        ax = pretty_plot(12, 8)
        for dos in self._doses.values():
            energies = dos['energies']
            densities = dos['densities']
            if not ys:
                ys = {Spin.up: np.zeros(energies.shape), Spin.down: np.zeros(energies.shape)}
            new_dens = {}
            for spin in [Spin.up, Spin.down]:
                if spin in densities:
                    if self.stack:
                        ys[spin] += densities[spin]
                        new_dens[spin] = ys[spin].copy()
                    else:
                        new_dens[spin] = densities[spin]
            all_energies.append(energies)
            all_densities.append(new_dens)
        keys = list(reversed(self._doses))
        all_densities.reverse()
        all_energies.reverse()
        all_pts = []
        for idx, key in enumerate(keys):
            for spin in [Spin.up, Spin.down]:
                if spin in all_densities[idx]:
                    energy = all_energies[idx]
                    densities = list(int(spin) * all_densities[idx][spin])
                    if invert_axes:
                        x = densities
                        y = energy
                    else:
                        x = energy
                        y = densities
                    all_pts.extend(list(zip(x, y)))
                    if self.stack:
                        ax.fill(x, y, color=colors[idx % n_colors], label=str(key))
                    elif spin == Spin.down and beta_dashed:
                        ax.plot(x, y, color=colors[idx % n_colors], label=str(key), linestyle='--', linewidth=3)
                    else:
                        ax.plot(x, y, color=colors[idx % n_colors], label=str(key), linewidth=3)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        elif not invert_axes:
            xlim = ax.get_xlim()
            relevant_y = [p[1] for p in all_pts if xlim[0] < p[0] < xlim[1]]
            ax.set_ylim((min(relevant_y), max(relevant_y)))
        if not xlim and invert_axes:
            ylim = ax.get_ylim()
            relevant_y = [p[0] for p in all_pts if ylim[0] < p[1] < ylim[1]]
            ax.set_xlim((min(relevant_y), max(relevant_y)))
        if self.zero_at_efermi:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(xlim, [0, 0], 'k--', linewidth=2) if invert_axes else ax.plot([0, 0], ylim, 'k--', linewidth=2)
        if invert_axes:
            ax.set_ylabel('Energies (eV)')
            ax.set_xlabel(f'Density of states (states/eV{('/Å³' if self._norm_val else '')})')
            ax.axvline(x=0, color='k', linestyle='--', linewidth=2)
        else:
            ax.set_xlabel('Energies (eV)')
            if self._norm_val:
                ax.set_ylabel('Density of states (states/eV/Å³)')
            else:
                ax.set_ylabel('Density of states (states/eV)')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
        handles, labels = ax.get_legend_handles_labels()
        label_dict = dict(zip(labels, handles))
        ax.legend(label_dict.values(), label_dict)
        legend_text = ax.get_legend().get_texts()
        plt.setp(legend_text, fontsize=30)
        plt.tight_layout()
        return ax

    def save_plot(self, filename: str, xlim=None, ylim=None, invert_axes=False, beta_dashed=False) -> None:
        """Save matplotlib plot to a file.

        Args:
            filename (str): Filename to write to. Must include extension to specify image format.
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes (bool): Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            beta_dashed (bool): Plots the beta spin channel with a dashed line. Defaults to False.
        """
        self.get_plot(xlim, ylim, invert_axes, beta_dashed)
        plt.savefig(filename)

    def show(self, xlim=None, ylim=None, invert_axes=False, beta_dashed=False) -> None:
        """Show the plot using matplotlib.

        Args:
            xlim: Specifies the x-axis limits. Set to None for automatic
                determination.
            ylim: Specifies the y-axis limits.
            invert_axes (bool): Whether to invert the x and y axes. Enables chemist style DOS plotting.
                Defaults to False.
            beta_dashed (bool): Plots the beta spin channel with a dashed line. Defaults to False.
        """
        self.get_plot(xlim, ylim, invert_axes, beta_dashed)
        plt.show()