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
class SurfaceEnergyPlotter:
    """
    A class used for generating plots to analyze the thermodynamics of surfaces
    of a material. Produces stability maps of different slab configurations,
    phases diagrams of two parameters to determine stability of configurations
    (future release), and Wulff shapes.

    Attributes:
        all_slab_entries (dict | list): Either a list of SlabEntry objects (note for a list, the
            SlabEntry must have the adsorbates and clean_entry parameter plugged in) or a Nested
            dictionary containing a list of entries for slab calculations as
            items and the corresponding Miller index of the slab as the key.
            To account for adsorption, each value is a sub-dictionary with the
            entry of a clean slab calculation as the sub-key and a list of
            entries for adsorption calculations as the sub-value. The sub-value
            can contain different adsorption configurations such as a different
            site or a different coverage, however, ordinarily only the most stable
            configuration for a particular coverage will be considered as the
            function of the adsorbed surface energy has an intercept dependent on
            the adsorption energy (ie an adsorption site with a higher adsorption
            energy will always provide a higher surface energy than a site with a
            lower adsorption energy). An example parameter is provided:
            {(h1,k1,l1): {clean_entry1: [ads_entry1, ads_entry2, ...], clean_entry2: [...], ...}, (h2,k2,l2): {...}}
            where clean_entry1 can be a pristine surface and clean_entry2 can be a
            reconstructed surface while ads_entry1 can be adsorption at site 1 with
            a 2x2 coverage while ads_entry2 can have a 3x3 coverage. If adsorption
            entries are present (i.e. if all_slab_entries[(h,k,l)][clean_entry1]), we
            consider adsorption in all plots and analysis for this particular facet.
        color_dict (dict): Dictionary of colors (r,g,b,a) when plotting surface energy stability.
            The keys are individual surface entries where clean surfaces have a solid color while
            the corresponding adsorbed surface will be transparent.
        ucell_entry (ComputedStructureEntry): ComputedStructureEntry of the bulk reference for
            this particular material.
        ref_entries (list): List of ComputedStructureEntries to be used for calculating chemical potential.
        facet_color_dict (dict): Randomly generated dictionary of colors associated with each facet.
    """

    def __init__(self, all_slab_entries, ucell_entry, ref_entries=None):
        """
        Object for plotting surface energy in different ways for clean and
            adsorbed surfaces.

        Args:
            all_slab_entries (dict or list): Dictionary or list containing
                all entries for slab calculations. See attributes.
            ucell_entry (ComputedStructureEntry): ComputedStructureEntry
                of the bulk reference for this particular material.
            ref_entries ([ComputedStructureEntries]): A list of entries for
                each type of element to be used as a reservoir for
                non-stoichiometric systems. The length of this list MUST be
                n-1 where n is the number of different elements in the bulk
                entry. The bulk energy term in the grand surface potential can
                be defined by a summation of the chemical potentials for each
                element in the system. As the bulk energy is already provided,
                one can solve for one of the chemical potentials as a function
                of the other chemical potentials and bulk energy. i.e. there
                are n-1 variables (chempots). e.g. if your ucell_entry is for
                LiFePO4 than your ref_entries should have an entry for Li, Fe,
                and P if you want to use the chempot of O as the variable.
        """
        self.ucell_entry = ucell_entry
        self.ref_entries = ref_entries
        self.all_slab_entries = all_slab_entries if type(all_slab_entries).__name__ == 'dict' else entry_dict_from_list(all_slab_entries)
        self.color_dict = self.color_palette_dict()
        se_dict, as_coeffs_dict = ({}, {})
        for hkl in self.all_slab_entries:
            for clean in self.all_slab_entries[hkl]:
                se = clean.surface_energy(self.ucell_entry, ref_entries=self.ref_entries)
                if type(se).__name__ == 'float':
                    se_dict[clean] = se
                    as_coeffs_dict[clean] = {1: se}
                else:
                    se_dict[clean] = se
                    as_coeffs_dict[clean] = se.as_coefficients_dict()
                for dope in self.all_slab_entries[hkl][clean]:
                    se = dope.surface_energy(self.ucell_entry, ref_entries=self.ref_entries)
                    if type(se).__name__ == 'float':
                        se_dict[dope] = se
                        as_coeffs_dict[dope] = {1: se}
                    else:
                        se_dict[dope] = se
                        as_coeffs_dict[dope] = se.as_coefficients_dict()
        self.surfe_dict = se_dict
        self.as_coeffs_dict = as_coeffs_dict
        list_of_chempots = []
        for v in self.as_coeffs_dict.values():
            if type(v).__name__ == 'float':
                continue
            for du in v:
                if du not in list_of_chempots:
                    list_of_chempots.append(du)
        self.list_of_chempots = list_of_chempots

    def get_stable_entry_at_u(self, miller_index, delu_dict=None, delu_default=0, no_doped=False, no_clean=False):
        """
        Returns the entry corresponding to the most stable slab for a particular
            facet at a specific chempot. We assume that surface energy is constant
            so all free variables must be set with delu_dict, otherwise they are
            assumed to be equal to delu_default.

        Args:
            miller_index ((h,k,l)): The facet to find the most stable slab in
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.

        Returns:
            SlabEntry, surface_energy (float)
        """
        all_delu_dict = self.set_all_variables(delu_dict, delu_default)

        def get_coeffs(e):
            coeffs = []
            for du in all_delu_dict:
                if type(self.as_coeffs_dict[e]).__name__ == 'float':
                    coeffs.append(self.as_coeffs_dict[e])
                elif du in self.as_coeffs_dict[e]:
                    coeffs.append(self.as_coeffs_dict[e][du])
                else:
                    coeffs.append(0)
            return np.array(coeffs)
        all_entries, all_coeffs = ([], [])
        for entry in self.all_slab_entries[miller_index]:
            if not no_clean:
                all_entries.append(entry)
                all_coeffs.append(get_coeffs(entry))
            if not no_doped:
                for ads_entry in self.all_slab_entries[miller_index][entry]:
                    all_entries.append(ads_entry)
                    all_coeffs.append(get_coeffs(ads_entry))
        du_vals = np.array(list(all_delu_dict.values()))
        all_gamma = list(np.dot(all_coeffs, du_vals.T))
        return (all_entries[all_gamma.index(min(all_gamma))], float(min(all_gamma)))

    def wulff_from_chempot(self, delu_dict=None, delu_default=0, symprec=1e-05, no_clean=False, no_doped=False):
        """
        Method to get the Wulff shape at a specific chemical potential.

        Args:
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            symprec (float): See WulffShape.
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.

        Returns:
            WulffShape: The WulffShape at u_ref and u_ads.
        """
        lattice = SpacegroupAnalyzer(self.ucell_entry.structure).get_conventional_standard_structure().lattice
        miller_list = list(self.all_slab_entries)
        e_surf_list = []
        for hkl in miller_list:
            gamma = self.get_stable_entry_at_u(hkl, delu_dict=delu_dict, delu_default=delu_default, no_clean=no_clean, no_doped=no_doped)[1]
            e_surf_list.append(gamma)
        return WulffShape(lattice, miller_list, e_surf_list, symprec=symprec)

    def area_frac_vs_chempot_plot(self, ref_delu: Symbol, chempot_range: list[float], delu_dict: dict[Symbol, float] | None=None, delu_default: float=0, increments: int=10, no_clean: bool=False, no_doped: bool=False) -> plt.Axes:
        """
        1D plot. Plots the change in the area contribution
        of each facet as a function of chemical potential.

        Args:
            ref_delu (Symbol): The free variable chempot with the format:
                Symbol("delu_el") where el is the name of the element.
            chempot_range (list[float]): Min/max range of chemical potential to plot along.
            delu_dict (dict[Symbol, float]): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials.
            increments (int): Number of data points between min/max or point
                of intersection. Defaults to 10 points.
            no_clean (bool): Some parameter, description missing.
            no_doped (bool): Some parameter, description missing.

        Returns:
            plt.Axes: Plot of area frac on the Wulff shape for each facet vs chemical potential.
        """
        delu_dict = delu_dict or {}
        chempot_range = sorted(chempot_range)
        all_chempots = np.linspace(min(chempot_range), max(chempot_range), increments)
        hkl_area_dict: dict[tuple[int, int, int], list[float]] = {}
        for hkl in self.all_slab_entries:
            hkl_area_dict[hkl] = []
        for u in all_chempots:
            delu_dict[ref_delu] = u
            wulff_shape = self.wulff_from_chempot(delu_dict=delu_dict, no_clean=no_clean, no_doped=no_doped, delu_default=delu_default)
            for hkl in wulff_shape.area_fraction_dict:
                hkl_area_dict[hkl].append(wulff_shape.area_fraction_dict[hkl])
        ax = pretty_plot(width=8, height=7)
        for hkl in self.all_slab_entries:
            clean_entry = next(iter(self.all_slab_entries[hkl]))
            if all((a == 0 for a in hkl_area_dict[hkl])):
                continue
            plt.plot(all_chempots, hkl_area_dict[hkl], '--', color=self.color_dict[clean_entry], label=str(hkl))
        ax.set(ylabel='Fractional area $A^{Wulff}_{hkl}/A^{Wulff}$')
        self.chempot_plot_addons(ax, chempot_range, str(ref_delu).split('_')[1], rect=[-0.0, 0, 0.95, 1], pad=5, ylim=[0, 1])
        return ax

    def get_surface_equilibrium(self, slab_entries, delu_dict=None):
        """
        Takes in a list of SlabEntries and calculates the chemical potentials
            at which all slabs in the list coexists simultaneously. Useful for
            building surface phase diagrams. Note that to solve for x equations
            (x slab_entries), there must be x free variables (chemical potentials).
            Adjust delu_dict as need be to get the correct number of free variables.

        Args:
            slab_entries (array): The coefficients of the first equation
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.

        Returns:
            array: Array containing a solution to x equations with x
                variables (x-1 chemical potential and 1 surface energy)
        """
        all_parameters = []
        all_eqns = []
        for slab_entry in slab_entries:
            se = self.surfe_dict[slab_entry]
            if type(se).__name__ == 'float':
                all_eqns.append(se - Symbol('gamma'))
            else:
                se = sub_chempots(se, delu_dict) if delu_dict else se
                all_eqns.append(se - Symbol('gamma'))
                all_parameters.extend([p for p in list(se.free_symbols) if p not in all_parameters])
        all_parameters.append(Symbol('gamma'))
        solution = linsolve(all_eqns, all_parameters)
        if not solution:
            warnings.warn('No solution')
            return solution
        return {param: next(iter(solution))[idx] for idx, param in enumerate(all_parameters)}

    def stable_u_range_dict(self, chempot_range, ref_delu, no_doped=True, no_clean=False, delu_dict=None, miller_index=(), dmu_at_0=False, return_se_dict=False):
        """
        Creates a dictionary where each entry is a key pointing to a
        chemical potential range where the surface of that entry is stable.
        Does so by enumerating through all possible solutions (intersect)
        for surface energies of a specific facet.

        Args:
            chempot_range ([max_chempot, min_chempot]): Range to consider the
                stability of the slabs.
            ref_delu (sympy Symbol): The range stability of each slab is based
                on the chempot range of this chempot. Should be a sympy Symbol
                object of the format: Symbol("delu_el") where el is the name of
                the element
            no_doped (bool): Consider stability of clean slabs only.
            no_clean (bool): Consider stability of doped slabs only.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            miller_index (list): Miller index for a specific facet to get a
                dictionary for.
            dmu_at_0 (bool): If True, if the surface energies corresponding to
                the chemical potential range is between a negative and positive
                value, the value is a list of three chemical potentials with the
                one in the center corresponding a surface energy of 0. Uselful
                in identifying unphysical ranges of surface energies and their
                chemical potential range.
            return_se_dict (bool): Whether or not to return the corresponding
                dictionary of surface energies
        """
        if delu_dict is None:
            delu_dict = {}
        chempot_range = sorted(chempot_range)
        stable_urange_dict, se_dict = ({}, {})
        for hkl in self.all_slab_entries:
            entries_in_hkl = []
            if miller_index and hkl != tuple(miller_index):
                continue
            if not no_clean:
                entries_in_hkl.extend(self.all_slab_entries[hkl])
            if not no_doped:
                for entry in self.all_slab_entries[hkl]:
                    entries_in_hkl.extend(self.all_slab_entries[hkl][entry])
            for entry in entries_in_hkl:
                stable_urange_dict[entry] = []
                se_dict[entry] = []
            if len(entries_in_hkl) == 1:
                stable_urange_dict[entries_in_hkl[0]] = chempot_range
                u1, u2 = (delu_dict.copy(), delu_dict.copy())
                u1[ref_delu], u2[ref_delu] = (chempot_range[0], chempot_range[1])
                se = self.as_coeffs_dict[entries_in_hkl[0]]
                se_dict[entries_in_hkl[0]] = [sub_chempots(se, u1), sub_chempots(se, u2)]
                continue
            for pair in itertools.combinations(entries_in_hkl, 2):
                solution = self.get_surface_equilibrium(pair, delu_dict=delu_dict)
                if not solution:
                    continue
                new_delu_dict = delu_dict.copy()
                new_delu_dict[ref_delu] = solution[ref_delu]
                stable_entry, gamma = self.get_stable_entry_at_u(hkl, new_delu_dict, no_doped=no_doped, no_clean=no_clean)
                if stable_entry not in pair:
                    continue
                if not chempot_range[0] <= solution[ref_delu] <= chempot_range[1]:
                    continue
                for entry in pair:
                    stable_urange_dict[entry].append(solution[ref_delu])
                    se_dict[entry].append(gamma)
            new_delu_dict = delu_dict.copy()
            for u in chempot_range:
                new_delu_dict[ref_delu] = u
                entry, gamma = self.get_stable_entry_at_u(hkl, delu_dict=new_delu_dict, no_doped=no_doped, no_clean=no_clean)
                stable_urange_dict[entry].append(u)
                se_dict[entry].append(gamma)
        if dmu_at_0:
            for entry, v in se_dict.items():
                if not stable_urange_dict[entry]:
                    continue
                if v[0] * v[1] < 0:
                    se = self.as_coeffs_dict[entry]
                    v.append(0)
                    stable_urange_dict[entry].append(solve(sub_chempots(se, delu_dict), ref_delu)[0])
        for entry, v in stable_urange_dict.items():
            se_dict[entry] = [se for idx, se in sorted(zip(v, se_dict[entry]))]
            stable_urange_dict[entry] = sorted(v)
        if return_se_dict:
            return (stable_urange_dict, se_dict)
        return stable_urange_dict

    def color_palette_dict(self, alpha=0.35):
        """
        Helper function to assign each facet a unique color using a dictionary.

        Args:
            alpha (float): Degree of transparency

        return (dict): Dictionary of colors (r,g,b,a) when plotting surface
            energy stability. The keys are individual surface entries where
            clean surfaces have a solid color while the corresponding adsorbed
            surface will be transparent.
        """
        color_dict = {}
        for hkl in self.all_slab_entries:
            rgb_indices = [0, 1, 2]
            color = [0, 0, 0, 1]
            random.shuffle(rgb_indices)
            for idx, ind in enumerate(rgb_indices):
                if idx == 2:
                    break
                color[ind] = np.random.uniform(0, 1)
            clean_list = np.linspace(0, 1, len(self.all_slab_entries[hkl]))
            for idx, clean in enumerate(self.all_slab_entries[hkl]):
                c = copy.copy(color)
                c[rgb_indices[2]] = clean_list[idx]
                color_dict[clean] = c
                for ads_entry in self.all_slab_entries[hkl][clean]:
                    c_ads = copy.copy(c)
                    c_ads[3] = alpha
                    color_dict[ads_entry] = c_ads
        return color_dict

    def chempot_vs_gamma_plot_one(self, ax: plt.Axes, entry: SlabEntry, ref_delu: Symbol, chempot_range: list[float], delu_dict: dict[Symbol, float] | None=None, delu_default: float=0, label: str='', JPERM2: bool=False) -> plt.Axes:
        """
        Helper function to help plot the surface energy of a
        single SlabEntry as a function of chemical potential.

        Args:
            ax (plt.Axes): Matplotlib Axes instance for plotting.
            entry: Entry of the slab whose surface energy we want
                to plot. (Add appropriate description for type)
            ref_delu (Symbol): The range stability of each slab is based
                on the chempot range of this chempot.
            chempot_range (list[float]): Range to consider the stability of the slabs.
            delu_dict (dict[Symbol, float]): Dictionary of the chemical potentials.
            delu_default (float): Default value for all unset chemical potentials.
            label (str): Label of the slab for the legend.
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False).

        Returns:
            plt.Axes: Plot of surface energy vs chemical potential for one entry.
        """
        delu_dict = delu_dict or {}
        chempot_range = sorted(chempot_range)
        ax = ax or plt.gca()
        ucell_comp = self.ucell_entry.composition.reduced_composition
        if entry.adsorbates:
            struct = entry.cleaned_up_slab
            clean_comp = struct.composition.reduced_composition
        else:
            clean_comp = entry.composition.reduced_composition
        mark = '--' if ucell_comp != clean_comp else '-'
        delu_dict = self.set_all_variables(delu_dict, delu_default)
        delu_dict[ref_delu] = chempot_range[0]
        gamma_min = self.as_coeffs_dict[entry]
        gamma_min = gamma_min if type(gamma_min).__name__ == 'float' else sub_chempots(gamma_min, delu_dict)
        delu_dict[ref_delu] = chempot_range[1]
        gamma_max = self.as_coeffs_dict[entry]
        gamma_max = gamma_max if type(gamma_max).__name__ == 'float' else sub_chempots(gamma_max, delu_dict)
        gamma_range = [gamma_min, gamma_max]
        se_range = np.array(gamma_range) * EV_PER_ANG2_TO_JOULES_PER_M2 if JPERM2 else gamma_range
        mark = entry.mark or mark
        color = entry.color or self.color_dict[entry]
        ax.plot(chempot_range, se_range, mark, color=color, label=label)
        return ax

    def chempot_vs_gamma(self, ref_delu, chempot_range, miller_index=(), delu_dict=None, delu_default=0, JPERM2=False, show_unstable=False, ylim=None, plt=None, no_clean=False, no_doped=False, use_entry_labels=False, no_label=False):
        """
        Plots the surface energy as a function of chemical potential.
            Each facet will be associated with its own distinct colors.
            Dashed lines will represent stoichiometries different from that
            of the mpid's compound. Transparent lines indicates adsorption.

        Args:
            ref_delu (sympy Symbol): The range stability of each slab is based
                on the chempot range of this chempot. Should be a sympy Symbol
                object of the format: Symbol("delu_el") where el is the name of
                the element
            chempot_range ([max_chempot, min_chempot]): Range to consider the
                stability of the slabs.
            miller_index (list): Miller index for a specific facet to get a
                dictionary for.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False)
            show_unstable (bool): Whether or not to show parts of the surface
                energy plot outside the region of stability.
            ylim ([ymax, ymin]): Range of y axis
            no_doped (bool): Whether to plot for the clean slabs only.
            no_clean (bool): Whether to plot for the doped slabs only.
            use_entry_labels (bool): If True, will label each slab configuration
                according to their given label in the SlabEntry object.
            no_label (bool): Option to turn off labels.

        Returns:
            Plot: Plot of surface energy vs chempot for all entries.
        """
        if delu_dict is None:
            delu_dict = {}
        chempot_range = sorted(chempot_range)
        plt = plt or pretty_plot(width=8, height=7)
        axes = plt.gca()
        for hkl in self.all_slab_entries:
            if miller_index and hkl != tuple(miller_index):
                continue
            if not show_unstable:
                stable_u_range_dict = self.stable_u_range_dict(chempot_range, ref_delu, no_doped=no_doped, delu_dict=delu_dict, miller_index=hkl)
            already_labelled = []
            label = ''
            for clean_entry in self.all_slab_entries[hkl]:
                urange = stable_u_range_dict[clean_entry] if not show_unstable else chempot_range
                if urange != []:
                    label = clean_entry.label
                    if label in already_labelled:
                        label = None
                    else:
                        already_labelled.append(label)
                    if not no_clean:
                        if use_entry_labels:
                            label = clean_entry.label
                        if no_label:
                            label = ''
                        plt = self.chempot_vs_gamma_plot_one(plt, clean_entry, ref_delu, urange, delu_dict=delu_dict, delu_default=delu_default, label=label, JPERM2=JPERM2)
                if not no_doped:
                    for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                        urange = stable_u_range_dict[ads_entry] if not show_unstable else chempot_range
                        if urange != []:
                            if use_entry_labels:
                                label = ads_entry.label
                            if no_label:
                                label = ''
                            plt = self.chempot_vs_gamma_plot_one(plt, ads_entry, ref_delu, urange, delu_dict=delu_dict, delu_default=delu_default, label=label, JPERM2=JPERM2)
        plt.ylabel('Surface energy (J/$m^{2}$)') if JPERM2 else plt.ylabel('Surface energy (eV/$\\AA^{2}$)')
        return self.chempot_plot_addons(plt, chempot_range, str(ref_delu).split('_')[1], axes, ylim=ylim)

    def monolayer_vs_BE(self, plot_eads=False):
        """
        Plots the binding energy as a function of monolayers (ML), i.e.
            the fractional area adsorbate density for all facets. For each
            facet at a specific monolayer, only plot the lowest binding energy.

        Args:
            plot_eads (bool): Option to plot the adsorption energy (binding
                energy multiplied by number of adsorbates) instead.

        Returns:
            Plot: Plot of binding energy vs monolayer for all facets.
        """
        ax = pretty_plot(width=8, height=7)
        for hkl in self.all_slab_entries:
            ml_be_dict = {}
            for clean_entry in self.all_slab_entries[hkl]:
                if self.all_slab_entries[hkl][clean_entry]:
                    for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                        if ads_entry.get_monolayer not in ml_be_dict:
                            ml_be_dict[ads_entry.get_monolayer] = 1000
                        be = ads_entry.gibbs_binding_energy(eads=plot_eads)
                        if be < ml_be_dict[ads_entry.get_monolayer]:
                            ml_be_dict[ads_entry.get_monolayer] = be
            vals = sorted(ml_be_dict.items())
            monolayers, BEs = zip(*vals)
            ax.plot(monolayers, BEs, '-o', c=self.color_dict[clean_entry], label=hkl)
        adsorbates = tuple(ads_entry.ads_entries_dict)
        ax.set_xlabel(f'{' '.join(adsorbates)} Coverage (ML)')
        ax.set_ylabel('Adsorption Energy (eV)' if plot_eads else 'Binding Energy (eV)')
        ax.legend()
        plt.tight_layout()
        return ax

    @staticmethod
    def chempot_plot_addons(ax, xrange, ref_el, pad=2.4, rect=None, ylim=None):
        """
        Helper function to a chempot plot look nicer.

        Args:
            plt (Plot) Plot to add things to.
            xrange (list): xlim parameter
            ref_el (str): Element of the referenced chempot.
            axes(axes) Axes object from matplotlib
            pad (float) For tight layout
            rect (list): For tight layout
            ylim (ylim parameter):

        return (Plot): Modified plot with addons.
        return (Plot): Modified plot with addons.
        """
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.0)
        ax.set_xlabel(f'Chemical potential $\\Delta\\mu_{{{ref_el}}}$ (eV)')
        ylim = ylim or ax.get_ylim()
        plt.xticks(rotation=60)
        plt.ylim(ylim)
        xlim = ax.get_xlim()
        plt.xlim(xlim)
        plt.tight_layout(pad=pad, rect=rect or [-0.047, 0, 0.84, 1])
        plt.plot([xrange[0], xrange[0]], ylim, '--k')
        plt.plot([xrange[1], xrange[1]], ylim, '--k')
        xy = [np.mean([xrange[1]]), np.mean(ylim)]
        plt.annotate(f'{ref_el}-rich', xy=xy, xytext=xy, rotation=90, fontsize=17)
        xy = [np.mean([xlim[0]]), np.mean(ylim)]
        plt.annotate(f'{ref_el}-poor', xy=xy, xytext=xy, rotation=90, fontsize=17)
        return ax

    def BE_vs_clean_SE(self, delu_dict, delu_default=0, plot_eads=False, annotate_monolayer=True, JPERM2=False):
        """
        For each facet, plot the clean surface energy against the most
            stable binding energy.

        Args:
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials
            plot_eads (bool): Option to plot the adsorption energy (binding
                energy multiplied by number of adsorbates) instead.
            annotate_monolayer (bool): Whether or not to label each data point
                with its monolayer (adsorbate density per unit primiitve area)
            JPERM2 (bool): Whether to plot surface energy in /m^2 (True) or
                eV/A^2 (False)

        Returns:
            Plot: Plot of clean surface energy vs binding energy for
                all facets.
        """
        ax = pretty_plot(width=8, height=7)
        for hkl in self.all_slab_entries:
            for clean_entry in self.all_slab_entries[hkl]:
                all_delu_dict = self.set_all_variables(delu_dict, delu_default)
                if self.all_slab_entries[hkl][clean_entry]:
                    clean_se = self.as_coeffs_dict[clean_entry]
                    se = sub_chempots(clean_se, all_delu_dict)
                    for ads_entry in self.all_slab_entries[hkl][clean_entry]:
                        ml = ads_entry.get_monolayer
                        be = ads_entry.gibbs_binding_energy(eads=plot_eads)
                        ax.scatter(se, be)
                        if annotate_monolayer:
                            ax.annotate(f'{ml:.2f}', xy=[se, be], xytext=[se, be])
        ax.set_xlabel('Surface energy ($J/m^2$)' if JPERM2 else 'Surface energy ($eV/\\AA^2$)')
        ax.set_ylabel('Adsorption Energy (eV)' if plot_eads else 'Binding Energy (eV)')
        plt.tight_layout()
        ax.set_xticks(rotation=60)
        return ax

    def surface_chempot_range_map(self, elements, miller_index, ranges, incr=50, no_doped=False, no_clean=False, delu_dict=None, ax=None, annotate=True, show_unphysical_only=False, fontsize=10) -> plt.Axes:
        """
        Adapted from the get_chempot_range_map() method in the PhaseDiagram
            class. Plot the chemical potential range map based on surface
            energy stability. Currently works only for 2-component PDs. At
            the moment uses a brute force method by enumerating through the
            range of the first element chempot with a specified increment
            and determines the chempot range of the second element for each
            SlabEntry. Future implementation will determine the chempot range
            map first by solving systems of equations up to 3 instead of 2.

        Args:
            elements (list): Sequence of elements to be considered as independent
                variables. E.g., if you want to show the stability ranges of
                all Li-Co-O phases w.r.t. to duLi and duO, you will supply
                [Element("Li"), Element("O")]
            miller_index ([h, k, l]): Miller index of the surface we are interested in
            ranges ([[range1], [range2]]): List of chempot ranges (max and min values)
                for the first and second element.
            incr (int): Number of points to sample along the range of the first chempot
            no_doped (bool): Whether or not to include doped systems.
            no_clean (bool): Whether or not to include clean systems.
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            ax (plt.Axes): Axes object to plot on. If None, will create a new plot.
            annotate (bool): Whether to annotate each "phase" with the label of
                the entry. If no label, uses the reduced formula
            show_unphysical_only (bool): Whether to only show the shaded region where
                surface energy is negative. Useful for drawing other chempot range maps.
            fontsize (int): Font size of the annotation
        """
        delu_dict = delu_dict or {}
        ax = ax or pretty_plot(12, 8)
        el1, el2 = (str(elements[0]), str(elements[1]))
        delu1 = Symbol(f'delu_{elements[0]}')
        delu2 = Symbol(f'delu_{elements[1]}')
        range1 = ranges[0]
        range2 = ranges[1]
        vertices_dict: dict[SlabEntry, list] = {}
        for dmu1 in np.linspace(range1[0], range1[1], incr):
            new_delu_dict = delu_dict.copy()
            new_delu_dict[delu1] = dmu1
            range_dict, se_dict = self.stable_u_range_dict(range2, delu2, dmu_at_0=True, miller_index=miller_index, no_doped=no_doped, no_clean=no_clean, delu_dict=new_delu_dict, return_se_dict=True)
            for entry, vertex in range_dict.items():
                if not vertex:
                    continue
                vertices_dict.setdefault(entry, [])
                selist = se_dict[entry]
                vertices_dict[entry].append({delu1: dmu1, delu2: [vertex, selist]})
        for entry, vertex in vertices_dict.items():
            xvals, yvals = ([], [])
            for ii, pt1 in enumerate(vertex):
                if len(pt1[delu2][1]) == 3:
                    if pt1[delu2][1][0] < 0:
                        neg_dmu_range = [pt1[delu2][0][0], pt1[delu2][0][1]]
                    else:
                        neg_dmu_range = [pt1[delu2][0][1], pt1[delu2][0][2]]
                    ax.plot([pt1[delu1], pt1[delu1]], neg_dmu_range, 'k--')
                elif pt1[delu2][1][0] < 0 and pt1[delu2][1][1] < 0 and (not show_unphysical_only):
                    ax.plot([pt1[delu1], pt1[delu1]], range2, 'k--')
                if ii == len(vertex) - 1:
                    break
                pt2 = vertex[ii + 1]
                if not show_unphysical_only:
                    ax.plot([pt1[delu1], pt2[delu1]], [pt1[delu2][0][0], pt2[delu2][0][0]], 'k')
                xvals.extend([pt1[delu1], pt2[delu1]])
                yvals.extend([pt1[delu2][0][0], pt2[delu2][0][0]])
            pt = vertex[-1]
            delu1, delu2 = pt
            xvals.extend([pt[delu1], pt[delu1]])
            yvals.extend(pt[delu2][0])
            if not show_unphysical_only:
                ax.plot([pt[delu1], pt[delu1]], [pt[delu2][0][0], pt[delu2][0][-1]], 'k')
            if annotate:
                x = np.mean([max(xvals), min(xvals)])
                y = np.mean([max(yvals), min(yvals)])
                label = entry.label or entry.reduced_formula
                ax.annotate(label, xy=[x, y], xytext=[x, y], fontsize=fontsize)
        ax.set(xlim=range1, ylim=range2)
        ax.set_xlabel(f'$\\Delta\\mu_{{{el1}}} (eV)$', fontsize=25)
        ax.set_ylabel(f'$\\Delta\\mu_{{{el2}}} (eV)$', fontsize=25)
        ax.set_xticks(rotation=60)
        return ax

    def set_all_variables(self, delu_dict, delu_default):
        """
        Sets all chemical potential values and returns a dictionary where
            the key is a sympy Symbol and the value is a float (chempot).

        Args:
            entry (SlabEntry): Computed structure entry of the slab
            delu_dict (dict): Dictionary of the chemical potentials to be set as
                constant. Note the key should be a sympy Symbol object of the
                format: Symbol("delu_el") where el is the name of the element.
            delu_default (float): Default value for all unset chemical potentials

        Returns:
            Dictionary of set chemical potential values
        """
        all_delu_dict = {}
        for du in self.list_of_chempots:
            if delu_dict and du in delu_dict:
                all_delu_dict[du] = delu_dict[du]
            elif du == 1:
                all_delu_dict[du] = du
            else:
                all_delu_dict[du] = delu_default
        return all_delu_dict