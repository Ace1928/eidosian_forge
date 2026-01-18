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
class MagneticStructureEnumerator:
    """Combines MagneticStructureAnalyzer and MagOrderingTransformation to
    automatically generate a set of transformations for a given structure
    and produce a list of plausible magnetic orderings.
    """
    available_strategies = ('ferromagnetic', 'antiferromagnetic', 'ferrimagnetic_by_motif', 'ferrimagnetic_by_species', 'antiferromagnetic_by_motif', 'nonmagnetic')

    def __init__(self, structure: Structure, default_magmoms: dict[str, float] | None=None, strategies: list[str] | tuple[str, ...]=('ferromagnetic', 'antiferromagnetic'), automatic: bool=True, truncate_by_symmetry: bool=True, transformation_kwargs: dict | None=None):
        """
        This class will try generated different collinear
        magnetic orderings for a given input structure.

        If the input structure has magnetic moments defined, it
        is possible to use these as a hint as to which elements are
        magnetic, otherwise magnetic elements will be guessed
        (this can be changed using default_magmoms kwarg).

        Args:
            structure: input structure
            default_magmoms: (optional, defaults provided) dict of
                magnetic elements to their initial magnetic moments in µB, generally
                these are chosen to be high-spin since they can relax to a low-spin
                configuration during a DFT electronic configuration
            strategies: different ordering strategies to use, choose from:
                ferromagnetic, antiferromagnetic, antiferromagnetic_by_motif,
                ferrimagnetic_by_motif and ferrimagnetic_by_species (here, "motif",
                means to use a different ordering parameter for symmetry inequivalent
                sites)
            automatic: if True, will automatically choose sensible strategies
            truncate_by_symmetry: if True, will remove very unsymmetrical
                orderings that are likely physically implausible
            transformation_kwargs: keyword arguments to pass to
                MagOrderingTransformation, to change automatic cell size limits, etc.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.structure = structure
        self.default_magmoms = default_magmoms
        self.strategies = list(strategies)
        self.automatic = automatic
        self.truncate_by_symmetry = truncate_by_symmetry
        self.num_orderings = 64
        self.max_unique_sites = 8
        default_transformation_kwargs = {'check_ordered_symmetry': False, 'timeout': 5}
        transformation_kwargs = transformation_kwargs or {}
        transformation_kwargs.update(default_transformation_kwargs)
        self.transformation_kwargs = transformation_kwargs
        self.ordered_structures: list[Structure] = []
        self.ordered_structure_origins: list[str] = []
        formula = structure.reduced_formula
        if not structure.is_ordered:
            raise ValueError(f'Please obtain an ordered approximation of the input structure ({formula}).')
        self.input_analyzer = CollinearMagneticStructureAnalyzer(structure, default_magmoms=default_magmoms, overwrite_magmom_mode='none')
        if not self.input_analyzer.is_collinear:
            raise ValueError(f'Input structure ({formula}) is non-collinear.')
        self.sanitized_structure = self._sanitize_input_structure(structure)
        self.transformations = self._generate_transformations(self.sanitized_structure)
        ordered_structures, ordered_structures_origins = self._generate_ordered_structures(self.sanitized_structure, self.transformations)
        self.ordered_structures = ordered_structures
        self.ordered_structure_origins = ordered_structures_origins

    @staticmethod
    def _sanitize_input_structure(struct: Structure) -> Structure:
        """Sanitize our input structure by removing magnetic information
        and making primitive.

        Args:
            struct: Structure

        Returns:
            Structure
        """
        struct = struct.copy()
        struct.remove_spin()
        struct = struct.get_primitive_structure(use_site_props=False)
        if 'magmom' in struct.site_properties:
            struct.remove_site_property('magmom')
        return struct

    def _generate_transformations(self, structure: Structure) -> dict[str, MagOrderingTransformation]:
        """The central problem with trying to enumerate magnetic orderings is
        that we have to enumerate orderings that might plausibly be magnetic
        ground states, while not enumerating orderings that are physically
        implausible. The problem is that it is not always obvious by e.g.
        symmetry arguments alone which orderings to prefer. Here, we use a
        variety of strategies (heuristics) to enumerate plausible orderings,
        and later discard any duplicates that might be found by multiple
        strategies. This approach is not ideal, but has been found to be
        relatively robust over a wide range of magnetic structures.

        Args:
            structure: A sanitized input structure (_sanitize_input_structure)

        Returns:
            dict: A dict of a transformation class instance (values) and name of enumeration strategy (keys)
        """
        formula = structure.reduced_formula
        transformations: dict[str, MagOrderingTransformation] = {}
        analyzer = CollinearMagneticStructureAnalyzer(structure, default_magmoms=self.default_magmoms, overwrite_magmom_mode='replace_all')
        if not analyzer.is_magnetic:
            raise ValueError('Not detected as magnetic, add a new default magmom for the element you believe may be magnetic?')
        self.logger.info(f'Generating magnetic orderings for {formula}')
        mag_species_spin = analyzer.magnetic_species_and_magmoms
        types_mag_species = sorted(analyzer.types_of_magnetic_species, key=lambda sp: analyzer.default_magmoms.get(str(sp), 0), reverse=True)
        num_mag_sites = analyzer.number_of_magnetic_sites
        num_unique_sites = analyzer.number_of_unique_magnetic_sites()
        if num_unique_sites > self.max_unique_sites:
            raise ValueError('Too many magnetic sites to sensibly perform enumeration.')
        if 'max_cell_size' not in self.transformation_kwargs:
            self.transformation_kwargs['max_cell_size'] = max(1, int(4 / num_mag_sites))
        self.logger.info(f'Max cell size set to {self.transformation_kwargs['max_cell_size']}')
        sga = SpacegroupAnalyzer(structure)
        structure_sym = sga.get_symmetrized_structure()
        wyckoff = ['n/a'] * len(structure)
        for indices, symbol in zip(structure_sym.equivalent_indices, structure_sym.wyckoff_symbols):
            for index in indices:
                wyckoff[index] = symbol
        is_magnetic_sites = [site.specie in types_mag_species for site in structure]
        wyckoff = [symbol if is_magnetic_site else 'n/a' for symbol, is_magnetic_site in zip(wyckoff, is_magnetic_sites)]
        structure.add_site_property('wyckoff', wyckoff)
        wyckoff_symbols = set(wyckoff) - {'n/a'}
        if self.automatic:
            if 'ferrimagnetic_by_motif' not in self.strategies and len(wyckoff_symbols) > 1 and (len(types_mag_species) == 1):
                self.strategies += ['ferrimagnetic_by_motif']
            if 'antiferromagnetic_by_motif' not in self.strategies and len(wyckoff_symbols) > 1 and (len(types_mag_species) == 1):
                self.strategies += ['antiferromagnetic_by_motif']
            if 'ferrimagnetic_by_species' not in self.strategies and len(types_mag_species) > 1:
                self.strategies += ['ferrimagnetic_by_species']
        if 'ferromagnetic' in self.strategies:
            fm_structure = analyzer.get_ferromagnetic_structure()
            fm_structure.add_spin_by_site(fm_structure.site_properties['magmom'])
            fm_structure.remove_site_property('magmom')
            self.ordered_structures.append(fm_structure)
            self.ordered_structure_origins.append('fm')
        all_constraints: dict[str, Any] = {}
        if 'antiferromagnetic' in self.strategies:
            constraint = MagOrderParameterConstraint(0.5, species_constraints=list(map(str, types_mag_species)))
            all_constraints['afm'] = [constraint]
            if len(types_mag_species) > 1:
                for sp in types_mag_species:
                    constraints = [MagOrderParameterConstraint(0.5, species_constraints=str(sp))]
                    all_constraints[f'afm_by_{sp}'] = constraints
        if 'ferrimagnetic_by_motif' in self.strategies and len(wyckoff_symbols) > 1:
            for symbol in wyckoff_symbols:
                constraints = [MagOrderParameterConstraint(0.5, site_constraint_name='wyckoff', site_constraints=symbol), MagOrderParameterConstraint(1, site_constraint_name='wyckoff', site_constraints=list(wyckoff_symbols - {symbol}))]
                all_constraints[f'ferri_by_motif_{symbol}'] = constraints
        if 'ferrimagnetic_by_species' in self.strategies:
            sp_list = [str(site.specie) for site in structure]
            num_sp = {sp: sp_list.count(str(sp)) for sp in types_mag_species}
            total_mag_sites = sum(num_sp.values())
            for sp in types_mag_species:
                all_constraints[f'ferri_by_{sp}'] = num_sp[sp] / total_mag_sites
                constraints = [MagOrderParameterConstraint(0.5, species_constraints=str(sp)), MagOrderParameterConstraint(1, species_constraints=list(map(str, set(types_mag_species) - {sp})))]
                all_constraints[f'ferri_by_{sp}_afm'] = constraints
        if 'antiferromagnetic_by_motif' in self.strategies:
            for symbol in wyckoff_symbols:
                constraints = [MagOrderParameterConstraint(0.5, site_constraint_name='wyckoff', site_constraints=symbol)]
                all_constraints[f'afm_by_motif_{symbol}'] = constraints
        transformations = {}
        for name, constraints in all_constraints.items():
            trans = MagOrderingTransformation(mag_species_spin, order_parameter=constraints, **self.transformation_kwargs)
            transformations[name] = trans
        return transformations

    def _generate_ordered_structures(self, sanitized_input_structure: Structure, transformations: dict[str, MagOrderingTransformation]) -> tuple[list[Structure], list[str]]:
        """Apply our input structure to our list of transformations and output a list
        of ordered structures that have been pruned for duplicates and for those
        with low symmetry (optional). Sets self.ordered_structures
        and self.ordered_structures_origins instance variables.

        Args:
            sanitized_input_structure: A sanitized input structure
            (_sanitize_input_structure)
            transformations: A dict of transformations (values) and name of
            enumeration strategy (key), the enumeration strategy name is just
            for record keeping


        Returns:
            list[Structures]
        """
        ordered_structures = self.ordered_structures
        ordered_structures_origins = self.ordered_structure_origins

        def _add_structures(ordered_structures, ordered_structures_origins, structures_to_add, origin=''):
            """Transformations with return_ranked_list can return either
            just Structures or dicts (or sometimes lists!) -- until this
            is fixed, we use this function to concat structures given
            by the transformation.
            """
            if structures_to_add:
                if isinstance(structures_to_add, Structure):
                    structures_to_add = [structures_to_add]
                structures_to_add = [s['structure'] if isinstance(s, dict) else s for s in structures_to_add]
                ordered_structures += structures_to_add
                ordered_structures_origins += [origin] * len(structures_to_add)
                self.logger.info(f'Adding {len(structures_to_add)} ordered structures: {origin}')
            return (ordered_structures, ordered_structures_origins)
        for origin, trans in self.transformations.items():
            structures_to_add = trans.apply_transformation(self.sanitized_structure, return_ranked_list=self.num_orderings)
            ordered_structures, ordered_structures_origins = _add_structures(ordered_structures, ordered_structures_origins, structures_to_add, origin=origin)
        self.logger.info('Pruning duplicate structures.')
        structures_to_remove: list[int] = []
        for idx, ordered_structure in enumerate(ordered_structures):
            if idx not in structures_to_remove:
                duplicate_checker = CollinearMagneticStructureAnalyzer(ordered_structure, overwrite_magmom_mode='none')
                for check_idx, check_structure in enumerate(ordered_structures):
                    if check_idx not in structures_to_remove and check_idx != idx and duplicate_checker.matches_ordering(check_structure):
                        structures_to_remove.append(check_idx)
        if len(structures_to_remove) == 0:
            self.logger.info(f'Removing {len(structures_to_remove)} duplicate ordered structures')
            ordered_structures = [s for idx, s in enumerate(ordered_structures) if idx not in structures_to_remove]
            ordered_structures_origins = [o for idx, o in enumerate(ordered_structures_origins) if idx not in structures_to_remove]
        if self.truncate_by_symmetry:
            if not isinstance(self.truncate_by_symmetry, int):
                self.truncate_by_symmetry = 5
            self.logger.info('Pruning low symmetry structures.')
            symmetry_int_numbers = [s.get_space_group_info()[1] for s in ordered_structures]
            num_sym_ops = [len(SpaceGroup.from_int_number(n).symmetry_ops) for n in symmetry_int_numbers]
            max_symmetries = sorted(set(num_sym_ops), reverse=True)
            if len(max_symmetries) > self.truncate_by_symmetry:
                max_symmetries = max_symmetries[0:5]
            structs_to_keep = [(idx, num) for idx, num in enumerate(num_sym_ops) if num in max_symmetries]
            structs_to_keep = sorted(structs_to_keep, key=lambda x: (x[1], -x[0]), reverse=True)
            self.logger.info(f'Removing {len(ordered_structures) - len(structs_to_keep)} low symmetry ordered structures')
            ordered_structures = [ordered_structures[idx] for idx, _struct in structs_to_keep]
            ordered_structures_origins = [ordered_structures_origins[idx] for idx, _struct in structs_to_keep]
            fm_index = ordered_structures_origins.index('fm')
            ordered_structures.insert(0, ordered_structures.pop(fm_index))
            ordered_structures_origins.insert(0, ordered_structures_origins.pop(fm_index))
        self.input_index = self.input_origin = None
        if self.input_analyzer.ordering != Ordering.NM:
            matches = [self.input_analyzer.matches_ordering(s) for s in ordered_structures]
            if not any(matches):
                ordered_structures.append(self.input_analyzer.structure)
                ordered_structures_origins.append('input')
                self.logger.info('Input structure not present in enumerated structures, adding...')
            else:
                self.logger.info(f'Input structure was found in enumerated structures at index {matches.index(True)}')
                self.input_index = matches.index(True)
                self.input_origin = ordered_structures_origins[self.input_index]
        return (ordered_structures, ordered_structures_origins)