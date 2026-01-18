from __future__ import annotations
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.serialization import loadfn
from ruamel.yaml.error import MarkedYAMLError
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
from pymatgen.core.structure import DummySpecies, Element, Species, Structure
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation, MagOrderParameterConstraint
from pymatgen.transformations.standard_transformations import AutoOxiStateDecorationTransformation
from pymatgen.util.due import Doi, due
class CollinearMagneticStructureAnalyzer:
    """
    A class which provides a few helpful methods to analyze
    collinear magnetic structures.
    """

    def __init__(self, structure: Structure, overwrite_magmom_mode: str | OverwriteMagmomMode=OverwriteMagmomMode.none, round_magmoms: bool=False, detect_valences: bool=False, make_primitive: bool=True, default_magmoms: dict | None=None, set_net_positive: bool=True, threshold: float=0, threshold_nonmag: float=0.1, threshold_ordering: float=1e-08):
        """
        If magnetic moments are not defined, moments will be
        taken either from default_magmoms.yaml (similar to the
        default magmoms in MPRelaxSet, with a few extra definitions)
        or from a specie:magmom dict provided by the default_magmoms
        kwarg.

        Input magmoms can be replaced using the 'overwrite_magmom_mode'
        kwarg. This can be:
        * "none" to do nothing,
        * "respect_sign" which will overwrite existing magmoms with
          those from default_magmoms but will keep sites with positive magmoms
          positive, negative magmoms negative and zero magmoms zero,
        * "respect_zeros", which will give a ferromagnetic structure
          (all positive magmoms from default_magmoms) but still keep sites with
          zero magmoms as zero,
        * "replace_all" which will try to guess initial magmoms for
          all sites in the structure irrespective of input structure
          (this is most suitable for an initial DFT calculation),
        * "replace_all_if_undefined" is the same as "replace_all" but only if
          no magmoms are defined in input structure, otherwise it will respect
          existing magmoms.
        * "normalize" will normalize magmoms to unity, but will respect sign
          (used for comparing orderings), magmoms < threshold will be set to zero

        Args:
            structure: input Structure object
            overwrite_magmom_mode: "respect_sign", "respect_zeros", "replace_all",
                "replace_all_if_undefined", "normalize" (default "none")
            round_magmoms: will round input magmoms to
                specified number of decimal places if integer is supplied, if set
                to a float will try and group magmoms together using a kernel density
                estimator of provided width, and extracting peaks of the estimator
                detect_valences: if True, will attempt to assign valences
                to input structure
            make_primitive: if True, will transform to primitive
                magnetic cell
            default_magmoms: (optional) dict specifying default magmoms
            set_net_positive: if True, will change sign of magnetic
                moments such that the net magnetization is positive. Argument will be
                ignored if mode "respect_sign" is used.
            threshold: number (in Bohr magnetons) below which magmoms
                will be rounded to zero
            threshold_nonmag: number (in Bohr magneton)
                below which nonmagnetic ions (with no magmom specified
                in default_magmoms) will be rounded to zero
            threshold_ordering: number (absolute of sum of all magmoms,
                in Bohr magneton) below which total magnetization is treated as zero
                when defining magnetic ordering. Defaults to 1e-8.
        """
        OverwriteMagmomMode(overwrite_magmom_mode)
        if default_magmoms:
            self.default_magmoms = default_magmoms
        else:
            self.default_magmoms = DEFAULT_MAGMOMS
        structure = structure.copy()
        if not structure.is_ordered:
            raise NotImplementedError(f'{type(self).__name__} not implemented for disordered structures, make ordered approximation first.')
        if detect_valences:
            trans = AutoOxiStateDecorationTransformation()
            try:
                structure = trans.apply_transformation(structure)
            except ValueError:
                warnings.warn(f'Could not assign valences for {structure.reduced_formula}')
        has_magmoms = bool(structure.site_properties.get('magmom', False))
        has_spin = False
        for comp in structure.species_and_occu:
            for sp in comp:
                if getattr(sp, 'spin', False):
                    has_spin = True
        if has_magmoms and has_spin:
            raise ValueError('Structure contains magnetic moments on both magmom site properties and spin species properties. This is ambiguous. Remove one or the other.')
        if has_magmoms:
            if None in structure.site_properties['magmom']:
                warnings.warn("Be careful with mixing types in your magmom site properties. Any 'None' magmoms have been replaced with zero.")
            magmoms = [m or 0 for m in structure.site_properties['magmom']]
        elif has_spin:
            magmoms = [sp.spin or 0 for sp in structure.species]
            structure.remove_spin()
        else:
            magmoms = [0] * len(structure)
            if overwrite_magmom_mode == 'replace_all_if_undefined':
                overwrite_magmom_mode = 'replace_all'
        self.is_collinear = Magmom.are_collinear(magmoms)
        if not self.is_collinear:
            warnings.warn('This class is not designed to be used with non-collinear structures. If your structure is only slightly non-collinear (e.g. canted) may still give useful results, but use with caution.')
        magmoms = list(map(float, magmoms))
        self.total_magmoms = sum(magmoms)
        self.magnetization = sum(magmoms) / structure.volume
        magmoms = [magmom if abs(magmom) > threshold and site.species_string in self.default_magmoms else magmom if abs(magmom) > threshold_nonmag and site.species_string not in self.default_magmoms else 0 for magmom, site in zip(magmoms, structure)]
        for idx, site in enumerate(structure):
            if site.species_string in self.default_magmoms:
                default_magmom = self.default_magmoms[site.species_string]
            elif isinstance(site.specie, Species) and str(site.specie.element) in self.default_magmoms:
                default_magmom = self.default_magmoms[str(site.specie.element)]
            else:
                default_magmom = 0
            if overwrite_magmom_mode == OverwriteMagmomMode.respect_sign.value:
                set_net_positive = False
                if magmoms[idx] > 0:
                    magmoms[idx] = default_magmom
                elif magmoms[idx] < 0:
                    magmoms[idx] = -default_magmom
            elif overwrite_magmom_mode == OverwriteMagmomMode.respect_zeros.value:
                if magmoms[idx] != 0:
                    magmoms[idx] = default_magmom
            elif overwrite_magmom_mode == OverwriteMagmomMode.replace_all.value:
                magmoms[idx] = default_magmom
            elif overwrite_magmom_mode == OverwriteMagmomMode.normalize.value and magmoms[idx] != 0:
                magmoms[idx] = int(magmoms[idx] / abs(magmoms[idx]))
        magmoms = self._round_magmoms(magmoms, round_magmoms) if round_magmoms else magmoms
        if set_net_positive:
            sign = np.sum(magmoms)
            if sign < 0:
                magmoms = [-x for x in magmoms]
        structure.add_site_property('magmom', magmoms)
        if make_primitive:
            structure = structure.get_primitive_structure(use_site_props=True)
        self.structure = structure
        self.threshold_ordering = threshold_ordering

    @no_type_check
    @staticmethod
    def _round_magmoms(magmoms: ArrayLike, round_magmoms_mode: float) -> np.ndarray:
        """If round_magmoms_mode is an integer, simply round to that number
        of decimal places, else if set to a float will try and round
        intelligently by grouping magmoms.
        """
        if isinstance(round_magmoms_mode, int):
            magmoms = np.around(magmoms, decimals=round_magmoms_mode)
        elif isinstance(round_magmoms_mode, float):
            try:
                range_m = max([max(magmoms), abs(min(magmoms))]) * 1.5
                kernel = gaussian_kde(magmoms, bw_method=round_magmoms_mode)
                x_grid = np.linspace(-range_m, range_m, int(1000 * range_m / round_magmoms_mode))
                kernel_m = kernel.evaluate(x_grid)
                extrema = x_grid[argrelextrema(kernel_m, comparator=np.greater)]
                magmoms = [extrema[np.abs(extrema - m).argmin()] for m in magmoms]
            except Exception as exc:
                warnings.warn('Failed to round magmoms intelligently, falling back to simple rounding.')
                warnings.warn(str(exc))
            n_decimals = len(str(round_magmoms_mode).split('.')[1]) + 1
            magmoms = np.around(magmoms, decimals=n_decimals)
        return np.array(magmoms)

    def get_structure_with_spin(self) -> Structure:
        """Returns a Structure with species decorated with spin values instead
        of using magmom site properties.
        """
        structure = self.structure.copy()
        structure.add_spin_by_site(structure.site_properties['magmom'])
        structure.remove_site_property('magmom')
        return structure

    def get_structure_with_only_magnetic_atoms(self, make_primitive: bool=True) -> Structure:
        """Returns a Structure with only magnetic atoms present.

        Args:
            make_primitive: Whether to make structure primitive after
                removing non-magnetic atoms (Default value = True)

        Returns:
            Structure
        """
        sites = [site for site in self.structure if abs(site.properties['magmom']) > 0]
        structure = Structure.from_sites(sites)
        if make_primitive:
            structure = structure.get_primitive_structure(use_site_props=True)
        return structure

    def get_nonmagnetic_structure(self, make_primitive: bool=True) -> Structure:
        """Returns a Structure without magnetic moments defined.

        Args:
            make_primitive: Whether to make structure primitive after
                removing magnetic information (Default value = True)

        Returns:
            Structure
        """
        structure = self.structure.copy()
        structure.remove_site_property('magmom')
        if make_primitive:
            structure = structure.get_primitive_structure()
        return structure

    def get_ferromagnetic_structure(self, make_primitive: bool=True) -> Structure:
        """Returns a Structure with all magnetic moments positive
        or zero.

        Args:
            make_primitive: Whether to make structure primitive after
                making all magnetic moments positive (Default value = True)

        Returns:
            Structure
        """
        structure = self.structure.copy()
        structure.add_site_property('magmom', [abs(m) for m in self.magmoms])
        if make_primitive:
            structure = structure.get_primitive_structure(use_site_props=True)
        return structure

    @property
    def is_magnetic(self) -> bool:
        """Convenience property, returns True if any non-zero magmoms present."""
        return any(map(abs, self.structure.site_properties['magmom']))

    @property
    def magmoms(self) -> np.ndarray:
        """Convenience property, returns magmoms as a numpy array."""
        return np.array(self.structure.site_properties['magmom'])

    @property
    def types_of_magnetic_species(self) -> tuple[Element | Species | DummySpecies, ...]:
        """Equivalent to Structure.types_of_specie but only returns magnetic species.

        Returns:
            tuple: types of Species
        """
        if self.number_of_magnetic_sites > 0:
            structure = self.get_structure_with_only_magnetic_atoms()
            return tuple(sorted(structure.types_of_species))
        return ()

    @property
    def types_of_magnetic_specie(self) -> tuple[Element | Species | DummySpecies, ...]:
        """Specie->Species rename. Used to maintain backwards compatibility."""
        return self.types_of_magnetic_species

    @property
    def magnetic_species_and_magmoms(self) -> dict[str, Any]:
        """Returns a dict of magnetic species and the magnitude of
        their associated magmoms. Will return a list if there are
        multiple magmoms per species.

        Returns:
            dict of magnetic species and magmoms
        """
        structure = self.get_ferromagnetic_structure()
        mag_types: dict = {str(site.specie): set() for site in structure if site.properties['magmom'] != 0}
        for site in structure:
            if site.properties['magmom'] != 0:
                mag_types[str(site.specie)].add(site.properties['magmom'])
        for sp, magmoms in mag_types.items():
            if len(magmoms) == 1:
                mag_types[sp] = magmoms.pop()
            else:
                mag_types[sp] = sorted(magmoms)
        return mag_types

    @property
    def number_of_magnetic_sites(self) -> int:
        """Number of magnetic sites present in structure."""
        return int(np.sum([abs(m) > 0 for m in self.magmoms]))

    def number_of_unique_magnetic_sites(self, symprec: float=0.001, angle_tolerance: float=5) -> int:
        """
        Args:
            symprec: same as in SpacegroupAnalyzer (Default value = 1e-3)
            angle_tolerance: same as in SpacegroupAnalyzer (Default value = 5).

        Returns:
            int: Number of symmetrically-distinct magnetic sites present in structure.
        """
        structure = self.get_nonmagnetic_structure()
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        symm_structure = sga.get_symmetrized_structure()
        num_unique_mag_sites = 0
        for group_of_sites in symm_structure.equivalent_sites:
            if group_of_sites[0].specie in self.types_of_magnetic_species:
                num_unique_mag_sites += 1
        return num_unique_mag_sites

    @property
    def ordering(self) -> Ordering:
        """Applies heuristics to return a magnetic ordering for a collinear
        magnetic structure. Result is not guaranteed to be correct, just a best
        guess. Tolerance for minimum total magnetization to be considered
        ferro/ferrimagnetic is self.threshold_ordering and defaults to 1e-8.

        Returns:
            Ordering: Enum  with values FM: ferromagnetic, FiM: ferrimagnetic,
                AFM: antiferromagnetic, NM: non-magnetic or Unknown. Unknown is
                returned if magnetic moments are not defined or structure is not collinear
                (in which case a warning is issued).
        """
        if not self.is_collinear:
            warnings.warn('Detecting ordering in non-collinear structures not yet implemented.')
            return Ordering.Unknown
        if 'magmom' not in self.structure.site_properties:
            return Ordering.Unknown
        magmoms = self.magmoms
        max_magmom = max(magmoms)
        total_magnetization = abs(sum(magmoms))
        is_potentially_ferromagnetic = np.all(magmoms >= 0) or np.all(magmoms <= 0)
        if abs(total_magnetization) > self.threshold_ordering and is_potentially_ferromagnetic:
            return Ordering.FM
        if abs(total_magnetization) > self.threshold_ordering:
            return Ordering.FiM
        if max_magmom > 0:
            return Ordering.AFM
        return Ordering.NM

    def get_exchange_group_info(self, symprec: float=0.01, angle_tolerance: float=5) -> tuple[str, int]:
        """Returns the information on the symmetry of the Hamiltonian
        describing the exchange energy of the system, taking into
        account relative direction of magnetic moments but not their
        absolute direction.

        This is not strictly accurate (e.g. some/many atoms will
        have zero magnetic moments), but defining symmetry this
        way is a useful way of keeping track of distinct magnetic
        orderings within pymatgen.

        Args:
            symprec: same as SpacegroupAnalyzer (Default value = 1e-2)
            angle_tolerance: same as SpacegroupAnalyzer (Default value = 5)

        Returns:
            spacegroup_symbol, international_number
        """
        structure = self.get_structure_with_spin()
        return structure.get_space_group_info(symprec=symprec, angle_tolerance=angle_tolerance)

    def matches_ordering(self, other: Structure) -> bool:
        """Compares the magnetic orderings of one structure with another.

        Args:
            other: Structure to compare

        Returns:
            bool: True if magnetic orderings match, False otherwise
        """
        cmag_analyzer = CollinearMagneticStructureAnalyzer(self.structure, overwrite_magmom_mode='normalize').get_structure_with_spin()
        b_positive = CollinearMagneticStructureAnalyzer(other, overwrite_magmom_mode='normalize', make_primitive=False)
        b_negative = b_positive.structure.copy()
        b_negative.add_site_property('magmom', -np.array(b_negative.site_properties['magmom']))
        analyzer = CollinearMagneticStructureAnalyzer(b_negative, overwrite_magmom_mode='normalize', make_primitive=False)
        b_positive = b_positive.get_structure_with_spin()
        analyzer = analyzer.get_structure_with_spin()
        return cmag_analyzer.matches(b_positive) or cmag_analyzer.matches(analyzer)

    def __str__(self):
        """
        Sorts a Structure (by fractional coordinate), and
        prints sites with magnetic information. This is
        useful over Structure.__str__ because sites are in
        a consistent order, which makes visual comparison between
        two identical Structures with different magnetic orderings
        easier.
        """
        frac_coords = self.structure.frac_coords
        sorted_indices = np.lexsort((frac_coords[:, 2], frac_coords[:, 1], frac_coords[:, 0]))
        struct = Structure.from_sites([self.structure[idx] for idx in sorted_indices])
        outs = ['Structure Summary', repr(struct.lattice)]
        outs.append('Magmoms Sites')
        for site in struct:
            prefix = f'{site.properties['magmom']:+.2f}   ' if site.properties['magmom'] != 0 else '        '
            outs.append(prefix + repr(site))
        return '\n'.join(outs)