from __future__ import annotations
import copy
import itertools
import random
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import linsolve, solve
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.surface import get_slab_regions
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot, Outcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
@due.dcite(Doi('10.1021/nl404557w'), description='Nanoscale stabilization of sodium oxides: Implications for Na-O2 batteries')
class NanoscaleStability:
    """A class for analyzing the stability of nanoparticles of different
    polymorphs with respect to size. The Wulff shape will be the model for the
    nanoparticle. Stability will be determined by an energetic competition between the
    weighted surface energy (surface energy of the Wulff shape) and the bulk energy. A
    future release will include a 2D phase diagram (e.g. w.r.t. size vs chempot for adsorbed
    or non-stoichiometric surfaces). Based on the following work:

    Kang, S., Mo, Y., Ong, S. P., & Ceder, G. (2014). Nanoscale
        stabilization of sodium oxides: Implications for Na-O2
        batteries. Nano Letters, 14(2), 1016-1020.
        https://doi.org/10.1021/nl404557w

    Attributes:
        se_analyzers (list[SurfaceEnergyPlotter]): Each item corresponds to a different polymorph.
        symprec (float): Tolerance for symmetry finding. See WulffShape.
    """

    def __init__(self, se_analyzers, symprec=1e-05):
        """Analyzes the nanoscale stability of different polymorphs."""
        self.se_analyzers = se_analyzers
        self.symprec = symprec

    def solve_equilibrium_point(self, analyzer1, analyzer2, delu_dict=None, delu_default=0, units='nanometers'):
        """
        Gives the radial size of two particles where equilibrium is reached
            between both particles. NOTE: the solution here is not the same
            as the solution visualized in the plot because solving for r
            requires that both the total surface area and volume of the
            particles are functions of r.

        Args:
            analyzer1 (SurfaceEnergyPlotter): Analyzer associated with the
                first polymorph
            analyzer2 (SurfaceEnergyPlotter): Analyzer associated with the
                second polymorph
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            units (str): Can be nanometers or Angstrom

        Returns:
            Particle radius in nm
        """
        wulff1 = analyzer1.wulff_from_chempot(delu_dict=delu_dict or {}, delu_default=delu_default, symprec=self.symprec)
        wulff2 = analyzer2.wulff_from_chempot(delu_dict=delu_dict or {}, delu_default=delu_default, symprec=self.symprec)
        delta_gamma = wulff1.weighted_surface_energy - wulff2.weighted_surface_energy
        delta_E = self.bulk_gform(analyzer1.ucell_entry) - self.bulk_gform(analyzer2.ucell_entry)
        radius = -3 * delta_gamma / delta_E
        return radius / 10 if units == 'nanometers' else radius

    def wulff_gform_and_r(self, wulff_shape, bulk_entry, r, from_sphere_area=False, r_units='nanometers', e_units='keV', normalize=False, scale_per_atom=False):
        """
        Calculates the formation energy of the particle with arbitrary radius r.

        Args:
            wulff_shape (WulffShape): Initial unscaled WulffShape
            bulk_entry (ComputedStructureEntry): Entry of the corresponding bulk.
            r (float (Ang)): Arbitrary effective radius of the WulffShape
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.
            r_units (str): Can be nanometers or Angstrom
            e_units (str): Can be keV or eV
            normalize (bool): Whether or not to normalize energy by volume
            scale_per_atom (True): Whether or not to normalize by number of
                atoms in the particle

        Returns:
            particle formation energy (float in keV), effective radius
        """
        miller_se_dict = wulff_shape.miller_energy_dict
        new_wulff = self.scaled_wulff(wulff_shape, r)
        new_wulff_area = new_wulff.miller_area_dict
        if not from_sphere_area:
            w_vol = new_wulff.volume
            tot_wulff_se = 0
            for hkl, v in new_wulff_area.items():
                tot_wulff_se += miller_se_dict[hkl] * v
            Ebulk = self.bulk_gform(bulk_entry) * w_vol
            new_r = new_wulff.effective_radius
        else:
            w_vol = 4 / 3 * np.pi * r ** 3
            sphere_sa = 4 * np.pi * r ** 2
            tot_wulff_se = wulff_shape.weighted_surface_energy * sphere_sa
            Ebulk = self.bulk_gform(bulk_entry) * w_vol
            new_r = r
        new_r = new_r / 10 if r_units == 'nanometers' else new_r
        e = Ebulk + tot_wulff_se
        e = e / 1000 if e_units == 'keV' else e
        e = e / (4 / 3 * np.pi * new_r ** 3) if normalize else e
        bulk_struct = bulk_entry.structure
        density = len(bulk_struct) / bulk_struct.volume
        e = e / (density * w_vol) if scale_per_atom else e
        return (e, new_r)

    @staticmethod
    def bulk_gform(bulk_entry):
        """
        Returns the formation energy of the bulk.

        Args:
            bulk_entry (ComputedStructureEntry): Entry of the corresponding bulk.

        Returns:
            float: bulk formation energy (in eV)
        """
        return bulk_entry.energy / bulk_entry.structure.volume

    def scaled_wulff(self, wulff_shape, r):
        """
        Scales the Wulff shape with an effective radius r. Note that the resulting
            Wulff does not necessarily have the same effective radius as the one
            provided. The Wulff shape is scaled by its surface energies where first
            the surface energies are scale by the minimum surface energy and then
            multiplied by the given effective radius.

        Args:
            wulff_shape (WulffShape): Initial, unscaled WulffShape
            r (float): Arbitrary effective radius of the WulffShape

        Returns:
            WulffShape (scaled by r)
        """
        r_ratio = r / wulff_shape.effective_radius
        miller_list = list(wulff_shape.miller_energy_dict)
        se_list = np.array(list(wulff_shape.miller_energy_dict.values()))
        scaled_se = se_list * r_ratio
        return WulffShape(wulff_shape.lattice, miller_list, scaled_se, symprec=self.symprec)

    def plot_one_stability_map(self, analyzer, max_r, delu_dict=None, label='', increments=50, delu_default=0, ax=None, from_sphere_area=False, e_units='keV', r_units='nanometers', normalize=False, scale_per_atom=False):
        """
        Returns the plot of the formation energy of a particle against its
            effect radius.

        Args:
            analyzer (SurfaceEnergyPlotter): Analyzer associated with the
                first polymorph
            max_r (float): The maximum radius of the particle to plot up to.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            label (str): Label of the plot for legend
            increments (int): Number of plot points
            delu_default (float): Default value for all unset chemical potentials
            plt (pyplot): Plot
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.
            r_units (str): Can be nanometers or Angstrom
            e_units (str): Can be keV or eV
            normalize (str): Whether or not to normalize energy by volume

        Returns:
            plt.Axes: matplotlib Axes object
        """
        ax = ax or pretty_plot(width=8, height=7)
        wulff_shape = analyzer.wulff_from_chempot(delu_dict=delu_dict, delu_default=delu_default, symprec=self.symprec)
        gform_list, r_list = ([], [])
        for radius in np.linspace(1e-06, max_r, increments):
            gform, radius = self.wulff_gform_and_r(wulff_shape, analyzer.ucell_entry, radius, from_sphere_area=from_sphere_area, r_units=r_units, e_units=e_units, normalize=normalize, scale_per_atom=scale_per_atom)
            gform_list.append(gform)
            r_list.append(radius)
        ru = 'nm' if r_units == 'nanometers' else '\\AA'
        ax.xlabel(f'Particle radius (${ru}$)')
        eu = f'${e_units}/{ru}^3$'
        ax.ylabel(f'$G_{{form}}$ ({eu})')
        ax.plot(r_list, gform_list, label=label)
        return ax

    def plot_all_stability_map(self, max_r, increments=50, delu_dict=None, delu_default=0, ax=None, labels=None, from_sphere_area=False, e_units='keV', r_units='nanometers', normalize=False, scale_per_atom=False):
        """
        Returns the plot of the formation energy of a particles
            of different polymorphs against its effect radius.

        Args:
            max_r (float): The maximum radius of the particle to plot up to.
            increments (int): Number of plot points
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            plt (pyplot): Plot
            labels (list): List of labels for each plot, corresponds to the
                list of se_analyzers
            from_sphere_area (bool): There are two ways to calculate the bulk
                formation energy. Either by treating the volume and thus surface
                area of the particle as a perfect sphere, or as a Wulff shape.

        Returns:
            plt.Axes: matplotlib Axes object
        """
        ax = ax or pretty_plot(width=8, height=7)
        for idx, analyzer in enumerate(self.se_analyzers):
            label = labels[idx] if labels else ''
            ax = self.plot_one_stability_map(analyzer, max_r, delu_dict, label=label, ax=ax, increments=increments, delu_default=delu_default, from_sphere_area=from_sphere_area, e_units=e_units, r_units=r_units, normalize=normalize, scale_per_atom=scale_per_atom)
        return ax