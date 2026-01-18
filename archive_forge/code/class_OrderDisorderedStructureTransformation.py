from __future__ import annotations
import logging
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from numpy import around
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, get_el_sp
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import PartialRemoveSitesTransformation
from pymatgen.transformations.transformation_abc import AbstractTransformation
class OrderDisorderedStructureTransformation(AbstractTransformation):
    """Order a disordered structure. The disordered structure must be oxidation
    state decorated for Ewald sum to be computed. No attempt is made to perform
    symmetry determination to reduce the number of combinations.

    Hence, attempting to order a large number of disordered sites can be extremely
    expensive. The time scales approximately with the
    number of possible combinations. The algorithm can currently compute
    approximately 5,000,000 permutations per minute.

    Also, simple rounding of the occupancies are performed, with no attempt
    made to achieve a target composition. This is usually not a problem for
    most ordering problems, but there can be times where rounding errors may
    result in structures that do not have the desired composition.
    This second step will be implemented in the next iteration of the code.

    If multiple fractions for a single species are found for different sites,
    these will be treated separately if the difference is above a threshold
    tolerance. currently this is .1

    For example, if a fraction of .25 Li is on sites 0, 1, 2, 3  and .5 on sites
    4, 5, 6, 7 then 1 site from [0, 1, 2, 3] will be filled and 2 sites from [4, 5, 6, 7]
    will be filled, even though a lower energy combination might be found by
    putting all lithium in sites [4, 5, 6, 7].

    USE WITH CARE.
    """
    ALGO_FAST = 0
    ALGO_COMPLETE = 1
    ALGO_BEST_FIRST = 2

    def __init__(self, algo=ALGO_FAST, symmetrized_structures=False, no_oxi_states=False):
        """
        Args:
            algo (int): Algorithm to use.
            symmetrized_structures (bool): Whether the input structures are
                instances of SymmetrizedStructure, and that their symmetry
                should be used for the grouping of sites.
            no_oxi_states (bool): Whether to remove oxidation states prior to
                ordering.
        """
        self.algo = algo
        self._all_structures = []
        self.no_oxi_states = no_oxi_states
        self.symmetrized_structures = symmetrized_structures

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False) -> Structure:
        """For this transformation, the apply_transformation method will return
        only the ordered structure with the lowest Ewald energy, to be
        consistent with the method signature of the other transformations.
        However, all structures are stored in the  all_structures attribute in
        the transformation object for easy access.

        Args:
            structure: Oxidation state decorated disordered structure to order
            return_ranked_list (bool | int, optional): If return_ranked_list is int, that number of structures
                is returned. If False, only the single lowest energy structure is returned. Defaults to False.

        Returns:
            Depending on returned_ranked list, either a transformed structure
            or a list of dictionaries, where each dictionary is of the form
            {"structure" = .... , "other_arguments"}
            the key "transformation" is reserved for the transformation that
            was actually applied to the structure.
            This transformation is parsed by the alchemy classes for generating
            a more specific transformation history. Any other information will
            be stored in the transformation_parameters dictionary in the
            transmuted structure class.
        """
        try:
            n_to_return = int(return_ranked_list)
        except ValueError:
            n_to_return = 1
        n_to_return = max(1, n_to_return)
        if self.no_oxi_states:
            structure = Structure.from_sites(structure)
            for idx, site in enumerate(structure):
                structure[idx] = {f'{k.symbol}0+': v for k, v in site.species.items()}
        equivalent_sites: list[list[int]] = []
        exemplars: list[PeriodicSite] = []
        for idx, site in enumerate(structure):
            if site.is_ordered:
                continue
            for j, ex in enumerate(exemplars):
                sp = ex.species
                if not site.species.almost_equals(sp):
                    continue
                if self.symmetrized_structures:
                    sym_equiv = structure.find_equivalent_sites(ex)
                    sym_test = site in sym_equiv
                else:
                    sym_test = True
                if sym_test:
                    equivalent_sites[j].append(idx)
                    break
            else:
                equivalent_sites.append([idx])
                exemplars.append(site)
        struct = Structure.from_sites(structure)
        manipulations = []
        for group in equivalent_sites:
            total_occupancy = dict(sum((structure[idx].species for idx in group), Composition()).items())
            for key, val in total_occupancy.items():
                if abs(val - round(val)) > 0.25:
                    raise ValueError('Occupancy fractions not consistent with size of unit cell')
                total_occupancy[key] = int(round(val))
            initial_sp = max(total_occupancy, key=lambda x: abs(x.oxi_state))
            for idx in group:
                struct[idx] = initial_sp
            for key, val in total_occupancy.items():
                if key == initial_sp:
                    continue
                oxi_ratio = key.oxi_state / initial_sp.oxi_state if initial_sp.oxi_state else 0
                manipulation = [oxi_ratio, val, list(group), key]
                manipulations.append(manipulation)
            empty = len(group) - sum(total_occupancy.values())
            if empty > 0.5:
                manipulations.append([0, empty, list(group), None])
        matrix = EwaldSummation(struct).total_energy_matrix
        ewald_m = EwaldMinimizer(matrix, manipulations, n_to_return, self.algo)
        self._all_structures = []
        lowest_energy = ewald_m.output_lists[0][0]
        n_atoms = sum(structure.composition.values())
        for output in ewald_m.output_lists:
            struct_copy = struct.copy()
            del_indices = []
            for manipulation in output[1]:
                if manipulation[1] is None:
                    del_indices.append(manipulation[0])
                else:
                    struct_copy[manipulation[0]] = manipulation[1]
            struct_copy.remove_sites(del_indices)
            if self.no_oxi_states:
                struct_copy.remove_oxidation_states()
            self._all_structures.append({'energy': output[0], 'energy_above_minimum': (output[0] - lowest_energy) / n_atoms, 'structure': struct_copy.get_sorted_structure()})
        if return_ranked_list:
            return self._all_structures[:n_to_return]
        return self._all_structures[0]['structure']

    def __repr__(self):
        return 'Order disordered structure transformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: True."""
        return True

    @property
    def lowest_energy_structure(self):
        """Lowest energy structure found."""
        return self._all_structures[0]['structure']