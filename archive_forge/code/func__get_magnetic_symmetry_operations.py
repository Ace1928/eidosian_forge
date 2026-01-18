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
def _get_magnetic_symmetry_operations(self, struct, grey_ops, atol):
    mag_ops = []
    magmoms = struct.site_properties['magmom']
    nonzero_magmom_inds = [idx for idx in range(len(struct)) if not (magmoms[idx] == np.array([0, 0, 0])).all()]
    init_magmoms = [site.properties['magmom'] for idx, site in enumerate(struct) if idx in nonzero_magmom_inds]
    sites = [site for idx, site in enumerate(struct) if idx in nonzero_magmom_inds]
    init_site_coords = [site.frac_coords for site in sites]
    for op in grey_ops:
        rot_mat = op.rotation_matrix
        t = op.translation_vector
        xformed_magmoms = [self._apply_op_to_magmom(rot_mat, magmom) for magmom in init_magmoms]
        xformed_site_coords = [np.dot(rot_mat, site.frac_coords) + t for site in sites]
        permutation = ['a' for i in range(len(sites))]
        not_found = list(range(len(sites)))
        for i in range(len(sites)):
            xformed = xformed_site_coords[i]
            for k, j in enumerate(not_found):
                init = init_site_coords[j]
                diff = xformed - init
                if self._all_ints(diff, atol=atol):
                    permutation[i] = j
                    not_found.pop(k)
                    break
        same = np.zeros(len(sites))
        flipped = np.zeros(len(sites))
        for i, magmom in enumerate(xformed_magmoms):
            if (magmom == init_magmoms[permutation[i]]).all():
                same[i] = 1
            elif (magmom == -1 * init_magmoms[permutation[i]]).all():
                flipped[i] = 1
        if same.all():
            mag_ops.append(MagSymmOp.from_rotation_and_translation_and_time_reversal(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector, time_reversal=1))
        if flipped.all():
            mag_ops.append(MagSymmOp.from_rotation_and_translation_and_time_reversal(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector, time_reversal=-1))
    return mag_ops