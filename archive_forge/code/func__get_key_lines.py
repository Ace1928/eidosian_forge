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
@staticmethod
def _get_key_lines(key_points, bz_as_key_point_inds):
    key_lines = []
    gamma_ind = len(key_points) - 1
    for facet_as_key_point_inds in bz_as_key_point_inds:
        facet_as_key_point_inds_bndy = facet_as_key_point_inds[:len(facet_as_key_point_inds) - 1]
        face_center_ind = facet_as_key_point_inds[-1]
        for j, ind in enumerate(facet_as_key_point_inds_bndy, start=-1):
            if (min(ind, facet_as_key_point_inds_bndy[j]), max(ind, facet_as_key_point_inds_bndy[j])) not in key_lines:
                key_lines.append((min(ind, facet_as_key_point_inds_bndy[j]), max(ind, facet_as_key_point_inds_bndy[j])))
            k = j + 2 if j != len(facet_as_key_point_inds_bndy) - 2 else 0
            if (min(ind, facet_as_key_point_inds_bndy[k]), max(ind, facet_as_key_point_inds_bndy[k])) not in key_lines:
                key_lines.append((min(ind, facet_as_key_point_inds_bndy[k]), max(ind, facet_as_key_point_inds_bndy[k])))
            if (ind, gamma_ind) not in key_lines:
                key_lines.append((ind, gamma_ind))
            key_lines.append((min(ind, face_center_ind), max(ind, face_center_ind)))
        key_lines.append((face_center_ind, gamma_ind))
    return key_lines