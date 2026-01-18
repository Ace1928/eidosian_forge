from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
def _get_klabels(self, lm_bs, sc_bs, hin_bs, rpg):
    """
        Returns:
            labels (dict): Dictionary of equivalent labels for paths if 'all' is chosen.
            If an exact kpoint match cannot be found, symmetric equivalency will be
            searched for and indicated with an asterisk in the equivalent label.
            If an equivalent label can still not be found, or the point is not in
            the explicit kpath, its equivalent label will be set to itself in the output.
        """
    lm_path = lm_bs.kpath
    sc_path = sc_bs.kpath
    hin_path = hin_bs.kpath
    n_op = len(rpg)
    pairs = itertools.permutations([{'setyawan_curtarolo': sc_path}, {'latimer_munro': lm_path}, {'hinuma': hin_path}], r=2)
    labels = {'setyawan_curtarolo': {}, 'latimer_munro': {}, 'hinuma': {}}
    for a, b in pairs:
        [(a_type, a_path)] = list(a.items())
        [(b_type, b_path)] = list(b.items())
        sc_count = np.zeros(n_op)
        for o_num in range(n_op):
            a_tr_coord = [np.dot(rpg[o_num], coord_a) for coord_a in a_path['kpoints'].values()]
            for coord_a in a_tr_coord:
                for value in b_path['kpoints'].values():
                    if np.allclose(value, coord_a, atol=self._atol):
                        sc_count[o_num] += 1
                        break
        a_to_b_labels = {}
        unlabeled = {}
        for label_a, coord_a in a_path['kpoints'].items():
            coord_a_t = np.dot(rpg[np.argmax(sc_count)], coord_a)
            assigned = False
            for label_b, coord_b in b_path['kpoints'].items():
                if np.allclose(coord_b, coord_a_t, atol=self._atol):
                    a_to_b_labels[label_a] = label_b
                    assigned = True
                    break
            if not assigned:
                unlabeled[label_a] = coord_a
        for label_a, coord_a in unlabeled.items():
            for op in rpg:
                coord_a_t = np.dot(op, coord_a)
                key = [key for key, value in b_path['kpoints'].items() if np.allclose(value, coord_a_t, atol=self._atol)]
                if key != []:
                    a_to_b_labels[label_a] = key[0][0] + '^{*}'
                    break
            if key == []:
                a_to_b_labels[label_a] = label_a
        labels[a_type][b_type] = a_to_b_labels
    return labels