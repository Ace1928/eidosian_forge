from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _get_rotation_reflection_matrices(self):
    """Compute candidates for the transformation matrix."""
    atoms1_ref = self._get_only_least_frequent_of(self.s1)
    cell = self.s1.get_cell().T
    cell_diag = np.sum(cell, axis=1)
    angle_tol = self.angle_tol
    delta_vec = 1e-06 * cell_diag
    ref_vec = self.s2.get_cell()
    ref_vec_lengths = np.linalg.norm(ref_vec, axis=1)
    ref_angles = np.array(self._get_angles(ref_vec))
    large_angles = ref_angles > np.pi / 2.0
    ref_angles[large_angles] = np.pi - ref_angles[large_angles]
    sc_atom_search = atoms1_ref * (3, 3, 3)
    new_sc_pos = sc_atom_search.get_positions()
    new_sc_pos -= new_sc_pos[0] + cell_diag - delta_vec
    lengths = np.linalg.norm(new_sc_pos, axis=1)
    candidate_indices = []
    rtol = self.ltol / len(self.s1)
    for k in range(3):
        correct_lengths_mask = np.isclose(lengths, ref_vec_lengths[k], rtol=rtol, atol=0)
        correct_lengths_mask[0] = False
        if not np.any(correct_lengths_mask):
            return None
        candidate_indices.append(np.nonzero(correct_lengths_mask)[0])
    aci = np.sort(list(set().union(*candidate_indices)))
    i2ang = dict(zip(aci, range(len(aci))))
    cosa = np.inner(new_sc_pos[aci], new_sc_pos[aci]) / np.outer(lengths[aci], lengths[aci])
    cosa[cosa > 1] = 1
    cosa[cosa < -1] = -1
    angles = np.arccos(cosa)
    angles[angles > np.pi / 2] = np.pi - angles[angles > np.pi / 2]
    refined_candidate_list = []
    for p in filterfalse(self._equal_elements_in_array, product(*candidate_indices)):
        a = np.array([angles[i2ang[p[0]], i2ang[p[1]]], angles[i2ang[p[0]], i2ang[p[2]]], angles[i2ang[p[1]], i2ang[p[2]]]])
        if np.allclose(a, ref_angles, atol=angle_tol, rtol=0):
            refined_candidate_list.append(new_sc_pos[np.array(p)].T)
    if len(refined_candidate_list) == 0:
        return None
    elif len(refined_candidate_list) == 1:
        inverted_trial = 1.0 / refined_candidate_list
    else:
        inverted_trial = np.linalg.inv(refined_candidate_list)
    candidate_trans_mat = np.dot(ref_vec.T, inverted_trial.T).T
    return candidate_trans_mat