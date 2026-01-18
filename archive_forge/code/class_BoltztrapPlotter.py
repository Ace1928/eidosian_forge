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
class BoltztrapPlotter:
    """class containing methods to plot the data from Boltztrap."""

    def __init__(self, bz) -> None:
        """
        Args:
            bz: a BoltztrapAnalyzer object.
        """
        self._bz = bz

    def _plot_doping(self, plt, temp) -> None:
        if len(self._bz.doping) != 0:
            limit = 2210000000000000.0
            plt.axvline(self._bz.mu_doping['n'][temp][0], linewidth=3.0, linestyle='--')
            plt.text(self._bz.mu_doping['n'][temp][0] + 0.01, limit, f'$n$=10^{{{math.log10(self._bz.doping['n'][0])}}}$', color='b')
            plt.axvline(self._bz.mu_doping['n'][temp][-1], linewidth=3.0, linestyle='--')
            plt.text(self._bz.mu_doping['n'][temp][-1] + 0.01, limit, f'$n$=10^{{{math.log10(self._bz.doping['n'][-1])}}}$', color='b')
            plt.axvline(self._bz.mu_doping['p'][temp][0], linewidth=3.0, linestyle='--')
            plt.text(self._bz.mu_doping['p'][temp][0] + 0.01, limit, f'$p$=10^{{{math.log10(self._bz.doping['p'][0])}}}$', color='b')
            plt.axvline(self._bz.mu_doping['p'][temp][-1], linewidth=3.0, linestyle='--')
            plt.text(self._bz.mu_doping['p'][temp][-1] + 0.01, limit, f'$p$=10^{{{math.log10(self._bz.doping['p'][-1])}}}$', color='b')

    def _plot_bg_limits(self, plt) -> None:
        plt.axvline(0.0, color='k', linewidth=3.0)
        plt.axvline(self._bz.gap, color='k', linewidth=3.0)

    def plot_seebeck_eff_mass_mu(self, temps=(300,), output='average', Lambda=0.5):
        """Plot respect to the chemical potential of the Seebeck effective mass
        calculated as explained in Ref.
        Gibbs, Z. M. et al., Effective mass and fermi surface complexity factor
        from ab initio band structure calculations.
        npj Computational Materials 3, 8 (2017).

        Args:
            output: 'average' returns the seebeck effective mass calculated
                using the average of the three diagonal components of the
                seebeck tensor. 'tensor' returns the seebeck effective mass
                respect to the three diagonal components of the seebeck tensor.
            temps: list of temperatures of calculated seebeck.
            Lambda: fitting parameter used to model the scattering (0.5 means
                constant relaxation time).

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        for temp in temps:
            sbk_mass = self._bz.get_seebeck_eff_mass(output=output, temp=temp, Lambda=0.5)
            start = self._bz.mu_doping['p'][temp][0]
            stop = self._bz.mu_doping['n'][temp][0]
            mu_steps_1 = []
            mu_steps_2 = []
            sbk_mass_1 = []
            sbk_mass_2 = []
            for i, mu in enumerate(self._bz.mu_steps):
                if mu <= start:
                    mu_steps_1.append(mu)
                    sbk_mass_1.append(sbk_mass[i])
                elif mu >= stop:
                    mu_steps_2.append(mu)
                    sbk_mass_2.append(sbk_mass[i])
            ax.plot(mu_steps_1, sbk_mass_1, label=f'{temp}K', linewidth=3)
            ax.plot(mu_steps_2, sbk_mass_2, linewidth=3.0)
            if output == 'average':
                ax.get_lines()[1].set_c(ax.get_lines()[0].get_c())
            elif output == 'tensor':
                ax.get_lines()[3].set_c(ax.get_lines()[0].get_c())
                ax.get_lines()[4].set_c(ax.get_lines()[1].get_c())
                ax.get_lines()[5].set_c(ax.get_lines()[2].get_c())
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
        ax.set_ylabel('Seebeck effective mass', fontsize=30)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        if output == 'tensor':
            ax.legend([f'{dim}_{T}K' for T in temps for dim in ('x', 'y', 'z')], fontsize=20)
        elif output == 'average':
            ax.legend(fontsize=20)
        plt.tight_layout()
        return ax

    def plot_complexity_factor_mu(self, temps=(300,), output='average', Lambda=0.5):
        """Plot respect to the chemical potential of the Fermi surface complexity
        factor calculated as explained in Ref.
        Gibbs, Z. M. et al., Effective mass and fermi surface complexity factor
        from ab initio band structure calculations.
        npj Computational Materials 3, 8 (2017).

        Args:
            output: 'average' returns the complexity factor calculated using the average
                of the three diagonal components of the seebeck and conductivity tensors.
                'tensor' returns the complexity factor respect to the three
                diagonal components of seebeck and conductivity tensors.
            temps: list of temperatures of calculated seebeck and conductivity.
            Lambda: fitting parameter used to model the scattering (0.5 means constant
                relaxation time).

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        for T in temps:
            cmplx_fact = self._bz.get_complexity_factor(output=output, temp=T, Lambda=Lambda)
            start = self._bz.mu_doping['p'][T][0]
            stop = self._bz.mu_doping['n'][T][0]
            mu_steps_1 = []
            mu_steps_2 = []
            cmplx_fact_1 = []
            cmplx_fact_2 = []
            for i, mu in enumerate(self._bz.mu_steps):
                if mu <= start:
                    mu_steps_1.append(mu)
                    cmplx_fact_1.append(cmplx_fact[i])
                elif mu >= stop:
                    mu_steps_2.append(mu)
                    cmplx_fact_2.append(cmplx_fact[i])
            ax.plot(mu_steps_1, cmplx_fact_1, label=str(T) + 'K', linewidth=3.0)
            ax.plot(mu_steps_2, cmplx_fact_2, linewidth=3.0)
            if output == 'average':
                ax.gca().get_lines()[1].set_c(ax.gca().get_lines()[0].get_c())
            elif output == 'tensor':
                ax.gca().get_lines()[3].set_c(ax.gca().get_lines()[0].get_c())
                ax.gca().get_lines()[4].set_c(ax.gca().get_lines()[1].get_c())
                ax.gca().get_lines()[5].set_c(ax.gca().get_lines()[2].get_c())
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
        ax.set_ylabel('Complexity Factor', fontsize=30)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        if output == 'tensor':
            ax.legend([f'{dim}_{T}K' for T in temps for dim in ('x', 'y', 'z')], fontsize=20)
        elif output == 'average':
            ax.legend(fontsize=20)
        plt.tight_layout()
        return ax

    def plot_seebeck_mu(self, temp: float=600, output: str='eig', xlim: Sequence[float] | None=None):
        """Plot the seebeck coefficient in function of Fermi level.

        Args:
            temp (float): the temperature
            output (str): "eig" or "average"
            xlim (tuple[float, float]): a 2-tuple of min and max fermi energy. Defaults to (0, band gap)

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        seebeck = self._bz.get_seebeck(output=output, doping_levels=False)[temp]
        ax.plot(self._bz.mu_steps, seebeck, linewidth=3.0)
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        if output == 'eig':
            ax.legend(['S$_1$', 'S$_2$', 'S$_3$'])
        if xlim is None:
            ax.set_xlim(-0.5, self._bz.gap + 0.5)
        else:
            ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylabel('Seebeck \n coefficient  ($\\mu$V/K)', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_conductivity_mu(self, temp: float=600, output: str='eig', relaxation_time: float=1e-14, xlim: Sequence[float] | None=None):
        """Plot the conductivity in function of Fermi level. Semi-log plot.

        Args:
            temp (float): the temperature
            output (str): "eig" or "average"
            relaxation_time (float): A relaxation time in s. Defaults to 1e-14 and the plot is in
               units of relaxation time
            xlim (tuple[float, float]): a 2-tuple of min and max fermi energy. Defaults to (0, band gap)

        Returns:
            a matplotlib object
        """
        cond = self._bz.get_conductivity(relaxation_time=relaxation_time, output=output, doping_levels=False)[temp]
        ax = pretty_plot(9, 7)
        ax.semilogy(self._bz.mu_steps, cond, linewidth=3.0)
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        if output == 'eig':
            ax.legend(['$\\Sigma_1$', '$\\Sigma_2$', '$\\Sigma_3$'])
        if xlim is None:
            ax.set_xlim(-0.5, self._bz.gap + 0.5)
        else:
            ax.set_xlim(xlim)
        ax.set_ylim([10000000000000.0 * relaxation_time, 1e+20 * relaxation_time])
        ax.set_ylabel('conductivity,\n $\\Sigma$ (1/($\\Omega$ m))', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30.0)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_power_factor_mu(self, temp: float=600, output: str='eig', relaxation_time: float=1e-14, xlim: Sequence[float] | None=None):
        """Plot the power factor in function of Fermi level. Semi-log plot.

        Args:
            temp (float): the temperature
            output (str): "eig" or "average"
            relaxation_time (float): A relaxation time in s. Defaults to 1e-14 and the plot is in
               units of relaxation time
            xlim (tuple[float, float]): a 2-tuple of min and max fermi energy. Defaults to (0, band gap)

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        pow_factor = self._bz.get_power_factor(relaxation_time=relaxation_time, output=output, doping_levels=False)[temp]
        ax.semilogy(self._bz.mu_steps, pow_factor, linewidth=3.0)
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        if output == 'eig':
            ax.legend(['PF$_1$', 'PF$_2$', 'PF$_3$'])
        if xlim is None:
            ax.set_xlim(-0.5, self._bz.gap + 0.5)
        else:
            ax.set_xlim(xlim)
        ax.set_ylabel('Power factor, ($\\mu$W/(mK$^2$))', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30.0)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_zt_mu(self, temp: float=600, output: str='eig', relaxation_time: float=1e-14, xlim: Sequence[float] | None=None) -> plt.Axes:
        """Plot the ZT as function of Fermi level.

        Args:
            temp (float): the temperature
            output (str): "eig" or "average"
            relaxation_time (float): A relaxation time in s. Defaults to 1e-14 and the plot is in
               units of relaxation time
            xlim (tuple[float, float]): a 2-tuple of min and max fermi energy. Defaults to (0, band gap)

        Returns:
            plt.Axes: matplotlib axes object
        """
        ax = pretty_plot(9, 7)
        zt = self._bz.get_zt(relaxation_time=relaxation_time, output=output, doping_levels=False)[temp]
        ax.plot(self._bz.mu_steps, zt, linewidth=3.0)
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        if output == 'eig':
            ax.legend(['ZT$_1$', 'ZT$_2$', 'ZT$_3$'])
        if xlim is None:
            ax.set_xlim(-0.5, self._bz.gap + 0.5)
        else:
            ax.set_xlim(xlim)
        ax.set_ylabel('ZT', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30.0)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_seebeck_temp(self, doping='all', output='average'):
        """Plot the Seebeck coefficient in function of temperature for different
        doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.

        Returns:
            a matplotlib object
        """
        if output == 'average':
            sbk = self._bz.get_seebeck(output='average')
        elif output == 'eigs':
            sbk = self._bz.get_seebeck(output='eigs')
        ax = pretty_plot(22, 14)
        tlist = sorted(sbk['n'])
        doping = self._bz.doping['n'] if doping == 'all' else doping
        for idx, doping_type in enumerate(['n', 'p']):
            plt.subplot(121 + idx)
            for dop in doping:
                dop_idx = self._bz.doping[doping_type].index(dop)
                sbk_temp = []
                for temp in tlist:
                    sbk_temp.append(sbk[doping_type][temp][dop_idx])
                if output == 'average':
                    ax.plot(tlist, sbk_temp, marker='s', label=f'{dop} $cm^{-3}$')
                elif output == 'eigs':
                    for xyz in range(3):
                        ax.plot(tlist, list(zip(*sbk_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
            ax.set_title(f'{doping_type}-type', fontsize=20)
            if idx == 0:
                ax.set_ylabel('Seebeck \n coefficient  ($\\mu$V/K)', fontsize=30.0)
            ax.set_xlabel('Temperature (K)', fontsize=30.0)
            ax.legend(loc='best', fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_conductivity_temp(self, doping='all', output='average', relaxation_time=1e-14):
        """Plot the conductivity in function of temperature for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            cond = self._bz.get_conductivity(relaxation_time=relaxation_time, output='average')
        elif output == 'eigs':
            cond = self._bz.get_conductivity(relaxation_time=relaxation_time, output='eigs')
        ax = pretty_plot(22, 14)
        tlist = sorted(cond['n'])
        doping = self._bz.doping['n'] if doping == 'all' else doping
        for idx, doping_type in enumerate(['n', 'p']):
            plt.subplot(121 + idx)
            for dop in doping:
                dop_idx = self._bz.doping[doping_type].index(dop)
                cond_temp = []
                for temp in tlist:
                    cond_temp.append(cond[doping_type][temp][dop_idx])
                if output == 'average':
                    ax.plot(tlist, cond_temp, marker='s', label=str(dop) + ' $cm^{-3}$')
                elif output == 'eigs':
                    for xyz in range(3):
                        ax.plot(tlist, list(zip(*cond_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
            ax.set_title(f'{doping_type}-type', fontsize=20)
            if idx == 0:
                ax.set_ylabel('conductivity $\\sigma$ (1/($\\Omega$ m))', fontsize=30.0)
            ax.set_xlabel('Temperature (K)', fontsize=30.0)
            ax.legend(loc='best', fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tight_layout()
        return ax

    def plot_power_factor_temp(self, doping='all', output='average', relaxation_time=1e-14):
        """Plot the Power Factor in function of temperature for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            pow_factor = self._bz.get_power_factor(relaxation_time=relaxation_time, output='average')
        elif output == 'eigs':
            pow_factor = self._bz.get_power_factor(relaxation_time=relaxation_time, output='eigs')
        ax = pretty_plot(22, 14)
        tlist = sorted(pow_factor['n'])
        doping = self._bz.doping['n'] if doping == 'all' else doping
        for idx, doping_type in enumerate(['n', 'p']):
            plt.subplot(121 + idx)
            for dop in doping:
                dop_idx = self._bz.doping[doping_type].index(dop)
                pf_temp = []
                for temp in tlist:
                    pf_temp.append(pow_factor[doping_type][temp][dop_idx])
                if output == 'average':
                    ax.plot(tlist, pf_temp, marker='s', label=f'{dop} $cm^{-3}$')
                elif output == 'eigs':
                    for xyz in range(3):
                        ax.plot(tlist, list(zip(*pf_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
            ax.set_title(f'{doping_type}-type', fontsize=20)
            if idx == 0:
                ax.set_ylabel('Power Factor ($\\mu$W/(mK$^2$))', fontsize=30.0)
            ax.set_xlabel('Temperature (K)', fontsize=30.0)
            ax.legend(loc='best', fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tight_layout()
        return ax

    def plot_zt_temp(self, doping='all', output: Literal['average', 'eigs']='average', relaxation_time=1e-14):
        """Plot the figure of merit zT in function of temperature for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Raises:
            ValueError: if output is not 'average' or 'eigs'

        Returns:
            a matplotlib object
        """
        if output not in ('average', 'eigs'):
            raise ValueError(f"output={output!r} must be 'average' or 'eigs'")
        zt = self._bz.get_zt(relaxation_time=relaxation_time, output=output)
        ax = pretty_plot(22, 14)
        tlist = sorted(zt['n'])
        doping = self._bz.doping['n'] if doping == 'all' else doping
        for idx, doping_type in enumerate(['n', 'p']):
            plt.subplot(121 + idx)
            for dop in doping:
                dop_idx = self._bz.doping[doping_type].index(dop)
                zt_temp = []
                for temp in tlist:
                    zt_temp.append(zt[doping_type][temp][dop_idx])
                if output == 'average':
                    ax.plot(tlist, zt_temp, marker='s', label=str(dop) + ' $cm^{-3}$')
                elif output == 'eigs':
                    for xyz in range(3):
                        ax.plot(tlist, list(zip(*zt_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
            ax.set_title(f'{doping_type}-type', fontsize=20)
            if idx == 0:
                ax.set_ylabel('zT', fontsize=30.0)
            ax.set_xlabel('Temperature (K)', fontsize=30.0)
            ax.legend(loc='best', fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_eff_mass_temp(self, doping='all', output: Literal['average', 'eigs']='average'):
        """Plot the average effective mass in function of temperature
        for different doping levels.

        Args:
            doping (str): the default 'all' plots all the doping levels in the analyzer.
                Specify a list of doping levels if you want to plot only some.
            output ('average' | 'eigs'): with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.

        Returns:
            a matplotlib Axes object
        """
        if output == 'average':
            eff_mass = self._bz.get_average_eff_mass(output='average')
        elif output == 'eigs':
            eff_mass = self._bz.get_average_eff_mass(output='eigs')
        ax_main = pretty_plot(22, 14)
        tlist = sorted(eff_mass['n'])
        doping = self._bz.doping['n'] if doping == 'all' else doping
        for idx, doping_type in enumerate(['n', 'p']):
            ax = plt.subplot(121 + idx)
            for dop in doping:
                dop_idx = self._bz.doping[doping_type].index(dop)
                em_temp = [eff_mass[doping_type][temp][dop_idx] for temp in tlist]
                if output == 'average':
                    ax.plot(tlist, em_temp, marker='s', label=f'{dop} $cm^{{-3}}$')
                elif output == 'eigs':
                    for xyz in range(3):
                        ax.plot(tlist, list(zip(*em_temp))[xyz], marker='s', label=f'{xyz} {dop} $cm^{{-3}}$')
            ax.set_title(f'{doping_type}-type', fontsize=20)
            if idx == 0:
                ax.set_ylabel('Effective mass (m$_e$)', fontsize=30.0)
            ax.set_xlabel('Temperature (K)', fontsize=30.0)
            ax.legend(loc='best', fontsize=15)
            ax.grid()
            ax.tick_params(labelsize=25)
        plt.tight_layout()
        return ax_main

    def plot_seebeck_dop(self, temps='all', output='average'):
        """Plot the Seebeck in function of doping levels for different temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.

        Returns:
            a matplotlib object
        """
        if output == 'average':
            sbk = self._bz.get_seebeck(output='average')
        elif output == 'eigs':
            sbk = self._bz.get_seebeck(output='eigs')
        tlist = sorted(sbk['n']) if temps == 'all' else temps
        ax = pretty_plot(22, 14)
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        ax.semilogx(self._bz.doping[dt], list(zip(*sbk[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
                elif output == 'average':
                    ax.semilogx(self._bz.doping[dt], sbk[dt][temp], marker='s', label=f'{temp} K')
            ax.set_title(dt + '-type', fontsize=20)
            if i == 0:
                ax.set_ylabel('Seebeck coefficient ($\\mu$V/K)', fontsize=30.0)
            ax.set_xlabel('Doping concentration (cm$^{-3}$)', fontsize=30.0)
            p = 'lower right' if i == 0 else 'best'
            ax.legend(loc=p, fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_conductivity_dop(self, temps='all', output='average', relaxation_time=1e-14):
        """Plot the conductivity in function of doping levels for different
        temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            cond = self._bz.get_conductivity(relaxation_time=relaxation_time, output='average')
        elif output == 'eigs':
            cond = self._bz.get_conductivity(relaxation_time=relaxation_time, output='eigs')
        tlist = sorted(cond['n']) if temps == 'all' else temps
        ax = pretty_plot(22, 14)
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        ax.semilogx(self._bz.doping[dt], list(zip(*cond[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
                elif output == 'average':
                    ax.semilogx(self._bz.doping[dt], cond[dt][temp], marker='s', label=f'{temp} K')
            ax.set_title(dt + '-type', fontsize=20)
            if i == 0:
                ax.set_ylabel('conductivity $\\sigma$ (1/($\\Omega$ m))', fontsize=30.0)
            ax.set_xlabel('Doping concentration ($cm^{-3}$)', fontsize=30.0)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.legend(fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_power_factor_dop(self, temps='all', output='average', relaxation_time=1e-14):
        """Plot the Power Factor in function of doping levels for different temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            pow_factor = self._bz.get_power_factor(relaxation_time=relaxation_time, output='average')
        elif output == 'eigs':
            pow_factor = self._bz.get_power_factor(relaxation_time=relaxation_time, output='eigs')
        tlist = sorted(pow_factor['n']) if temps == 'all' else temps
        ax = pretty_plot(22, 14)
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        ax.semilogx(self._bz.doping[dt], list(zip(*pow_factor[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
                elif output == 'average':
                    ax.semilogx(self._bz.doping[dt], pow_factor[dt][temp], marker='s', label=f'{temp} K')
            ax.set_title(dt + '-type', fontsize=20)
            if i == 0:
                ax.set_ylabel('Power Factor  ($\\mu$W/(mK$^2$))', fontsize=30.0)
            ax.set_xlabel('Doping concentration ($cm^{-3}$)', fontsize=30.0)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            p = 'best'
            ax.legend(loc=p, fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_zt_dop(self, temps='all', output='average', relaxation_time=1e-14):
        """Plot the figure of merit zT in function of doping levels for different
        temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            zt = self._bz.get_zt(relaxation_time=relaxation_time, output='average')
        elif output == 'eigs':
            zt = self._bz.get_zt(relaxation_time=relaxation_time, output='eigs')
        tlist = sorted(zt['n']) if temps == 'all' else temps
        ax = pretty_plot(22, 14)
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        ax.semilogx(self._bz.doping[dt], list(zip(*zt[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
                elif output == 'average':
                    ax.semilogx(self._bz.doping[dt], zt[dt][temp], marker='s', label=f'{temp} K')
            ax.set_title(dt + '-type', fontsize=20)
            if i == 0:
                ax.set_ylabel('zT', fontsize=30.0)
            ax.set_xlabel('Doping concentration ($cm^{-3}$)', fontsize=30.0)
            p = 'lower right' if i == 0 else 'best'
            ax.legend(loc=p, fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_eff_mass_dop(self, temps='all', output='average'):
        """Plot the average effective mass in function of doping levels
        for different temperatures.

        Args:
            temps: the default 'all' plots all the temperatures in the analyzer.
                Specify a list of temperatures if you want to plot only some.
            output: with 'average' you get an average of the three directions
                with 'eigs' you get all the three directions.
            relaxation_time: specify a constant relaxation time value

        Returns:
            a matplotlib object
        """
        if output == 'average':
            em = self._bz.get_average_eff_mass(output='average')
        elif output == 'eigs':
            em = self._bz.get_average_eff_mass(output='eigs')
        tlist = sorted(em['n']) if temps == 'all' else temps
        ax = pretty_plot(22, 14)
        for i, dt in enumerate(['n', 'p']):
            plt.subplot(121 + i)
            for temp in tlist:
                if output == 'eigs':
                    for xyz in range(3):
                        ax.semilogx(self._bz.doping[dt], list(zip(*em[dt][temp]))[xyz], marker='s', label=f'{xyz} {temp} K')
                elif output == 'average':
                    ax.semilogx(self._bz.doping[dt], em[dt][temp], marker='s', label=f'{temp} K')
            ax.set_title(dt + '-type', fontsize=20)
            if i == 0:
                ax.set_ylabel('Effective mass (m$_e$)', fontsize=30.0)
            ax.set_xlabel('Doping concentration ($cm^{-3}$)', fontsize=30.0)
            p = 'lower right' if i == 0 else 'best'
            ax.legend(loc=p, fontsize=15)
            ax.grid()
            ax.set_xticks(fontsize=25)
            ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_dos(self, sigma=0.05):
        """Plot dos.

        Args:
            sigma: a smearing

        Returns:
            a matplotlib object
        """
        plotter = DosPlotter(sigma=sigma)
        plotter.add_dos('t', self._bz.dos)
        return plotter.get_plot()

    def plot_carriers(self, temp=300):
        """Plot the carrier concentration in function of Fermi level.

        Args:
            temp: the temperature

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        carriers = [abs(c / (self._bz.vol * 1e-24)) for c in self._bz._carrier_conc[temp]]
        ax.semilogy(self._bz.mu_steps, carriers, linewidth=3.0, color='r')
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        ax.set_xlim(-0.5, self._bz.gap + 0.5)
        ax.set_ylim(100000000000000.0, 1e+22)
        ax.set_ylabel('carrier concentration (cm-3)', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax

    def plot_hall_carriers(self, temp=300):
        """Plot the Hall carrier concentration in function of Fermi level.

        Args:
            temp: the temperature

        Returns:
            a matplotlib object
        """
        ax = pretty_plot(9, 7)
        hall_carriers = [abs(i) for i in self._bz.get_hall_carrier_concentration()[temp]]
        ax.semilogy(self._bz.mu_steps, hall_carriers, linewidth=3.0, color='r')
        self._plot_bg_limits(ax)
        self._plot_doping(ax, temp)
        ax.set_xlim(-0.5, self._bz.gap + 0.5)
        ax.set_ylim(100000000000000.0, 1e+22)
        ax.set_ylabel('Hall carrier concentration (cm-3)', fontsize=30.0)
        ax.set_xlabel('E-E$_f$ (eV)', fontsize=30)
        ax.set_xticks(fontsize=25)
        ax.set_yticks(fontsize=25)
        plt.tight_layout()
        return ax