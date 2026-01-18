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