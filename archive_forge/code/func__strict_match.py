from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
def _strict_match(self, struct1: Structure, struct2: Structure, fu: int, s1_supercell: bool=True, use_rms: bool=False, break_on_match: bool=False) -> tuple[float, float, np.ndarray, float, Mapping] | None:
    """
        Matches struct2 onto struct1 (which should contain all sites in
        struct2).

        Args:
            struct1 (Structure): structure to match onto
            struct2 (Structure): structure to match
            fu (int): size of supercell to create
            s1_supercell (bool): whether to create the supercell of struct1 (vs struct2)
            use_rms (bool): whether to minimize the rms of the matching
            break_on_match (bool): whether to stop search at first match

        Returns:
            tuple[float, float, np.ndarray, float, Mapping]: (rms, max_dist, mask, cost, mapping)
                if a match is found, else None
        """
    if fu < 1:
        raise ValueError('fu cannot be less than 1')
    mask, s1_t_inds, s2_t_ind = self._get_mask(struct1, struct2, fu, s1_supercell)
    if mask.shape[0] > mask.shape[1]:
        raise ValueError('after supercell creation, struct1 must have more sites than struct2')
    if not self._subset and mask.shape[1] != mask.shape[0]:
        return None
    if LinearAssignment(mask).min_cost > 0:
        return None
    best_match = None
    for s1fc, s2fc, avg_l, sc_m in self._get_supercells(struct1, struct2, fu, s1_supercell):
        normalization = (len(s1fc) / avg_l.volume) ** (1 / 3)
        inv_abc = np.array(avg_l.reciprocal_lattice.abc)
        frac_tol = inv_abc * self.stol / (np.pi * normalization)
        for s1i in s1_t_inds:
            t = s1fc[s1i] - s2fc[s2_t_ind]
            t_s2fc = s2fc + t
            if self._cmp_fstruct(s1fc, t_s2fc, frac_tol, mask):
                inv_lll_abc = np.array(avg_l.get_lll_reduced_lattice().reciprocal_lattice.abc)
                lll_frac_tol = inv_lll_abc * self.stol / (np.pi * normalization)
                dist, t_adj, mapping = self._cart_dists(s1fc, t_s2fc, avg_l, mask, normalization, lll_frac_tol)
                val = np.linalg.norm(dist) / len(dist) ** 0.5 if use_rms else max(dist)
                if best_match is None or val < best_match[0]:
                    total_t = t + t_adj
                    total_t -= np.round(total_t)
                    best_match = (val, dist, sc_m, total_t, mapping)
                    if (break_on_match or val < 1e-05) and val < self.stol:
                        return best_match
    if best_match and best_match[0] < self.stol:
        return best_match
    return None