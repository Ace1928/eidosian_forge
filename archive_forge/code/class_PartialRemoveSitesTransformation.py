from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
class PartialRemoveSitesTransformation(AbstractTransformation):
    """Remove fraction of specie from a structure.
    Requires an oxidation state decorated structure for Ewald sum to be
    computed.

    Given that the solution to selecting the right removals is NP-hard, there
    are several algorithms provided with varying degrees of accuracy and speed.
    The options are as follows:

    ALGO_FAST:
        This is a highly optimized algorithm to quickly go through the search
        tree. It is guaranteed to find the optimal solution, but will return
        only a single lowest energy structure. Typically, you will want to use
        this.

    ALGO_COMPLETE:
        The complete algo ensures that you get all symmetrically distinct
        orderings, ranked by the estimated Ewald energy. But this can be an
        extremely time-consuming process if the number of possible orderings is
        very large. Use this if you really want all possible orderings. If you
        want just the lowest energy ordering, ALGO_FAST is accurate and faster.

    ALGO_BEST_FIRST:
        This algorithm is for ordering the really large cells that defeats even
        ALGO_FAST. For example, if you have 48 sites of which you want to
        remove 16 of them, the number of possible orderings is around
        2 x 10^12. ALGO_BEST_FIRST shortcircuits the entire search tree by
        removing the highest energy site first, then followed by the next
        highest energy site, and so on. It is guaranteed to find a solution
        in a reasonable time, but it is also likely to be highly inaccurate.

    ALGO_ENUMERATE:
        This algorithm uses the EnumerateStructureTransformation to perform
        ordering. This algo returns *complete* orderings up to a single unit
        cell size. It is more robust than the ALGO_COMPLETE, but requires
        Gus Hart's enumlib to be installed.
    """
    ALGO_FAST = 0
    ALGO_COMPLETE = 1
    ALGO_BEST_FIRST = 2
    ALGO_ENUMERATE = 3

    def __init__(self, indices, fractions, algo=ALGO_COMPLETE):
        """
        Args:
            indices:
                A list of list of indices, e.g. [[0, 1], [2, 3, 4, 5]].
            fractions:
                The corresponding fractions to remove. Must be same length as
                indices. e.g., [0.5, 0.25]
            algo:
                This parameter allows you to choose the algorithm to perform
                ordering. Use one of PartialRemoveSpecieTransformation.ALGO_*
                variables to set the algo.
        """
        self.indices = indices
        self.fractions = fractions
        self.algo = algo
        self.logger = logging.getLogger(type(self).__name__)

    def _best_first_ordering(self, structure: Structure, num_remove_dict):
        self.logger.debug('Performing best first ordering')
        start_time = time.perf_counter()
        self.logger.debug('Performing initial Ewald sum...')
        ewald_sum = EwaldSummation(structure)
        self.logger.debug(f'Ewald sum took {time.perf_counter() - start_time} seconds.')
        start_time = time.perf_counter()
        e_matrix = ewald_sum.total_energy_matrix
        to_delete = []
        total_removals = sum(num_remove_dict.values())
        removed = dict.fromkeys(num_remove_dict, 0)
        for _ in range(total_removals):
            max_idx = None
            max_ene = float('-inf')
            max_indices = None
            for indices in num_remove_dict:
                if removed[indices] < num_remove_dict[indices]:
                    for ind in indices:
                        if ind not in to_delete:
                            energy = sum(e_matrix[:, ind]) + sum(e_matrix[:, ind]) - e_matrix[ind, ind]
                            if energy > max_ene:
                                max_idx = ind
                                max_ene = energy
                                max_indices = indices
            removed[max_indices] += 1
            to_delete.append(max_idx)
            e_matrix[:, max_idx] = 0
            e_matrix[max_idx, :] = 0
        struct = structure.copy()
        struct.remove_sites(to_delete)
        self.logger.debug(f'Minimizing Ewald took {time.perf_counter() - start_time} seconds.')
        return [{'energy': sum(e_matrix), 'structure': struct.get_sorted_structure()}]

    def _complete_ordering(self, structure: Structure, num_remove_dict):
        self.logger.debug('Performing complete ordering...')
        all_structures: list[dict[str, float | Structure]] = []
        symprec = 0.2
        spg_analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        self.logger.debug(f'Symmetry of structure is determined to be {spg_analyzer.get_space_group_symbol()}.')
        sg = spg_analyzer.get_space_group_operations()
        tested_sites: list[list[PeriodicSite]] = []
        start_time = time.perf_counter()
        self.logger.debug('Performing initial Ewald sum...')
        ewald_sum = EwaldSummation(structure)
        self.logger.debug(f'Ewald sum took {time.perf_counter() - start_time} seconds.')
        start_time = time.perf_counter()
        all_combis = [list(itertools.combinations(ind, num)) for ind, num in num_remove_dict.items()]
        for idx, all_indices in enumerate(itertools.product(*all_combis), 1):
            sites_to_remove = []
            indices_list = []
            for indices in all_indices:
                sites_to_remove.extend([structure[i] for i in indices])
                indices_list.extend(indices)
            s_new = structure.copy()
            s_new.remove_sites(indices_list)
            energy = ewald_sum.compute_partial_energy(indices_list)
            already_tested = False
            for ii, t_sites in enumerate(tested_sites):
                t_energy = all_structures[ii]['energy']
                if abs((energy - t_energy) / len(s_new)) < 1e-05 and sg.are_symmetrically_equivalent(sites_to_remove, t_sites, symm_prec=symprec):
                    already_tested = True
            if not already_tested:
                tested_sites.append(sites_to_remove)
                all_structures.append({'structure': s_new, 'energy': energy})
            if idx % 10 == 0:
                now = time.perf_counter()
                self.logger.debug(f'{idx} structures, {now - start_time:.2f} seconds.')
                self.logger.debug(f'Average time per combi = {(now - start_time) / idx} seconds')
                self.logger.debug(f'{len(all_structures)} symmetrically distinct structures found.')
        self.logger.debug(f'Total symmetrically distinct structures found = {len(all_structures)}')
        return sorted(all_structures, key=lambda s: s['energy'])

    def _fast_ordering(self, structure: Structure, num_remove_dict, num_to_return=1):
        """This method uses the matrix form of Ewald sum to calculate the ewald
        sums of the potential structures. This is on the order of 4 orders of
        magnitude faster when there are large numbers of permutations to
        consider. There are further optimizations possible (doing a smarter
        search of permutations for example), but this won't make a difference
        until the number of permutations is on the order of 30,000.
        """
        self.logger.debug('Performing fast ordering')
        start_time = time.perf_counter()
        self.logger.debug('Performing initial Ewald sum...')
        ewald_matrix = EwaldSummation(structure).total_energy_matrix
        self.logger.debug(f'Ewald sum took {time.perf_counter() - start_time} seconds.')
        start_time = time.perf_counter()
        m_list = [[0, num, list(indices), None] for indices, num in num_remove_dict.items()]
        self.logger.debug('Calling EwaldMinimizer...')
        minimizer = EwaldMinimizer(ewald_matrix, m_list, num_to_return, PartialRemoveSitesTransformation.ALGO_FAST)
        self.logger.debug(f'Minimizing Ewald took {time.perf_counter() - start_time} seconds.')
        all_structures = []
        lowest_energy = minimizer.output_lists[0][0]
        num_atoms = sum(structure.composition.values())
        for output in minimizer.output_lists:
            struct = structure.copy()
            del_indices = []
            for manipulation in output[1]:
                if manipulation[1] is None:
                    del_indices.append(manipulation[0])
                else:
                    struct.replace(manipulation[0], manipulation[1])
            struct.remove_sites(del_indices)
            struct = struct.get_sorted_structure()
            e_above_min = (output[0] - lowest_energy) / num_atoms
            all_structures.append({'energy': output[0], 'energy_above_minimum': e_above_min, 'structure': struct})
        return all_structures

    def _enumerate_ordering(self, structure: Structure):
        struct = structure.copy()
        for indices, fraction in zip(self.indices, self.fractions):
            for ind in indices:
                new_sp = {sp: occu * fraction for sp, occu in structure[ind].species.items()}
                struct[ind] = new_sp
        from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
        trans = EnumerateStructureTransformation()
        return trans.apply_transformation(struct, 10000)

    def apply_transformation(self, structure: Structure, return_ranked_list: bool | int=False):
        """Apply the transformation.

        Args:
            structure: input structure
            return_ranked_list (bool | int): Whether or not multiple structures are returned.
                If return_ranked_list is int, that number of structures is returned.

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
        num_remove_dict = {}
        total_combos = 0
        for idx, frac in zip(self.indices, self.fractions):
            n_to_remove = len(idx) * frac
            if abs(n_to_remove - int(round(n_to_remove))) > 0.001:
                raise ValueError('Fraction to remove must be consistent with integer amounts in structure.')
            n_to_remove = int(round(n_to_remove))
            num_remove_dict[tuple(idx)] = n_to_remove
            n = len(idx)
            total_combos += int(round(math.factorial(n) / math.factorial(n_to_remove) / math.factorial(n - n_to_remove)))
        self.logger.debug(f'Total combinations = {total_combos}')
        try:
            num_to_return = int(return_ranked_list)
        except ValueError:
            num_to_return = 1
        num_to_return = max(1, num_to_return)
        self.logger.debug(f'Will return {num_to_return} best structures.')
        if self.algo == PartialRemoveSitesTransformation.ALGO_FAST:
            all_structures = self._fast_ordering(structure, num_remove_dict, num_to_return)
        elif self.algo == PartialRemoveSitesTransformation.ALGO_COMPLETE:
            all_structures = self._complete_ordering(structure, num_remove_dict)
        elif self.algo == PartialRemoveSitesTransformation.ALGO_BEST_FIRST:
            all_structures = self._best_first_ordering(structure, num_remove_dict)
        elif self.algo == PartialRemoveSitesTransformation.ALGO_ENUMERATE:
            all_structures = self._enumerate_ordering(structure)
        else:
            raise ValueError('Invalid algo.')
        opt_s = all_structures[0]['structure']
        return opt_s if not return_ranked_list else all_structures[0:num_to_return]

    def __repr__(self):
        return f'PartialRemoveSitesTransformation : Indices and fraction to remove = {self.indices}, ALGO = {self.algo}'

    @property
    def inverse(self) -> None:
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns True."""
        return True