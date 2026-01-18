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
@classmethod
def _cart_dists(cls, s1, s2, avg_lattice, mask, normalization, lll_frac_tol=None):
    """
        Finds a matching in Cartesian space. Finds an additional
        fractional translation vector to minimize RMS distance.

        Args:
            s1: numpy array of fractional coordinates.
            s2: numpy array of fractional coordinates. len(s1) >= len(s2)
            avg_lattice: Lattice on which to calculate distances
            mask: numpy array of booleans. mask[i, j] = True indicates
                that s2[i] cannot be matched to s1[j]
            normalization (float): inverse normalization length
            lll_frac_tol (float): tolerance for Lenstra-Lenstra-Lovász lattice basis reduction algorithm

        Returns:
            Distances from s2 to s1, normalized by (V/atom) ^ 1/3
            Fractional translation vector to apply to s2.
            Mapping from s1 to s2, i.e. with numpy slicing, s1[mapping] => s2
        """
    if len(s2) > len(s1):
        raise ValueError(f'len(s1)={len(s1)!r} must be larger than len(s2)={len(s2)!r}')
    if mask.shape != (len(s2), len(s1)):
        raise ValueError('mask has incorrect shape')
    vecs, d_2 = pbc_shortest_vectors(avg_lattice, s2, s1, mask, return_d2=True, lll_frac_tol=lll_frac_tol)
    lin = LinearAssignment(d_2)
    sol = lin.solution
    short_vecs = vecs[np.arange(len(sol)), sol]
    translation = np.average(short_vecs, axis=0)
    f_translation = avg_lattice.get_fractional_coords(translation)
    new_d2 = np.sum((short_vecs - translation) ** 2, axis=-1)
    return (new_d2 ** 0.5 * normalization, f_translation, sol)