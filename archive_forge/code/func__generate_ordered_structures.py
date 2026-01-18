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