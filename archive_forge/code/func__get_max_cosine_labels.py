from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
def _get_max_cosine_labels(self, max_cosine_orbits_orig, key_points_inds_orbits, atol):
    max_cosine_orbits_copy = max_cosine_orbits_orig.copy()
    max_cosine_label_inds = np.zeros(len(max_cosine_orbits_copy))
    initial_max_cosine_label_inds = [max_cos_orb[0][0] for max_cos_orb in max_cosine_orbits_copy]
    _uniq_vals, inds, counts = np.unique(initial_max_cosine_label_inds, return_index=True, return_counts=True)
    grouped_inds = [[i for i in range(len(initial_max_cosine_label_inds)) if max_cosine_orbits_copy[i][0][0] == max_cosine_orbits_copy[ind][0][0]] for ind in inds]
    pop_orbits = []
    pop_labels = []
    unassigned_orbits = []
    for idx, ind in enumerate(inds):
        if counts[idx] == 1:
            max_cosine_label_inds[ind] = initial_max_cosine_label_inds[ind]
            pop_orbits.append(ind)
            pop_labels.append(initial_max_cosine_label_inds[ind])
        else:
            next_choices = []
            for grouped_ind in grouped_inds[idx]:
                j = 1
                while True:
                    if max_cosine_orbits_copy[grouped_ind][j][0] not in initial_max_cosine_label_inds:
                        next_choices.append(max_cosine_orbits_copy[grouped_ind][j][1])
                        break
                    j += 1
            worst_next_choice = next_choices.index(min(next_choices))
            for grouped_ind in grouped_inds[idx]:
                if grouped_ind != worst_next_choice:
                    unassigned_orbits.append(grouped_ind)
            max_cosine_label_inds[grouped_inds[idx][worst_next_choice]] = initial_max_cosine_label_inds[grouped_inds[idx][worst_next_choice]]
            pop_orbits.append(grouped_inds[idx][worst_next_choice])
            pop_labels.append(initial_max_cosine_label_inds[grouped_inds[idx][worst_next_choice]])
    if unassigned_orbits:
        max_cosine_orbits_copy = self._reduce_cosines_array(max_cosine_orbits_copy, pop_orbits, pop_labels)
        unassigned_orbits_labels = self._get_orbit_labels(max_cosine_orbits_copy, key_points_inds_orbits, atol)
        for idx, unassigned_orbit in enumerate(unassigned_orbits):
            max_cosine_label_inds[unassigned_orbit] = unassigned_orbits_labels[idx]
    return max_cosine_label_inds