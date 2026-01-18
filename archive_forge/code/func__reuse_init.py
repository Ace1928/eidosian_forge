from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def _reuse_init(self, eigendisplacements: ArrayLike, frequencies: ArrayLike, has_nac: bool, qpoints: Sequence[Kpoint]) -> None:
    self.distance = []
    self.branches = []
    one_group: list = []
    branches_tmp = []
    previous_qpoint = self.qpoints[0]
    previous_distance = 0.0
    previous_label = self.qpoints[0].label
    for idx in range(self.nb_qpoints):
        label = self.qpoints[idx].label
        if label is not None and previous_label is not None:
            self.distance += [previous_distance]
        else:
            self.distance += [np.linalg.norm(self.qpoints[idx].cart_coords - previous_qpoint.cart_coords) + previous_distance]
        previous_qpoint = self.qpoints[idx]
        previous_distance = self.distance[idx]
        if label and previous_label:
            if len(one_group) != 0:
                branches_tmp += [one_group]
            one_group = []
        previous_label = label
        one_group += [idx]
    if len(one_group) != 0:
        branches_tmp += [one_group]
    for branch in branches_tmp:
        self.branches += [{'start_index': branch[0], 'end_index': branch[-1], 'name': f'{self.qpoints[branch[0]].label}-{self.qpoints[branch[-1]].label}'}]
    if has_nac:
        naf = []
        nac_eigendisplacements = []
        for idx in range(self.nb_qpoints):
            if np.allclose(qpoints[idx], (0, 0, 0)):
                if idx > 0 and (not np.allclose(qpoints[idx - 1], (0, 0, 0))):
                    q_dir = self.qpoints[idx - 1]
                    direction = q_dir.frac_coords / np.linalg.norm(q_dir.frac_coords)
                    naf.append((direction, frequencies[:, idx]))
                    if self.has_eigendisplacements:
                        nac_eigendisplacements.append((direction, eigendisplacements[:, idx]))
                if idx < len(qpoints) - 1 and (not np.allclose(qpoints[idx + 1], (0, 0, 0))):
                    q_dir = self.qpoints[idx + 1]
                    direction = q_dir.frac_coords / np.linalg.norm(q_dir.frac_coords)
                    naf.append((direction, frequencies[:, idx]))
                    if self.has_eigendisplacements:
                        nac_eigendisplacements.append((direction, eigendisplacements[:, idx]))
        self.nac_frequencies = np.array(naf, dtype=object)
        self.nac_eigendisplacements = np.array(nac_eigendisplacements, dtype=object)