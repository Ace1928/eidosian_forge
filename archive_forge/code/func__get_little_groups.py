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
def _get_little_groups(self, key_points, key_points_inds_orbits, key_lines_inds_orbits):
    little_groups_points = []
    for i, orbit in enumerate(key_points_inds_orbits):
        k0 = key_points[orbit[0]]
        little_groups_points.append([])
        for j, op in enumerate(self._rpg):
            gamma_to = np.dot(op, -1 * k0) + k0
            check_gamma = True
            if not self._all_ints(gamma_to, atol=self._atol):
                check_gamma = False
            if check_gamma:
                little_groups_points[i].append(j)
    little_groups_lines = []
    for i, orbit in enumerate(key_lines_inds_orbits):
        l0 = orbit[0]
        v = key_points[l0[1]] - key_points[l0[0]]
        k0 = key_points[l0[0]] + np.e / pi * v
        little_groups_lines.append([])
        for j, op in enumerate(self._rpg):
            gamma_to = np.dot(op, -1 * k0) + k0
            check_gamma = True
            if not self._all_ints(gamma_to, atol=self._atol):
                check_gamma = False
            if check_gamma:
                little_groups_lines[i].append(j)
    return (little_groups_points, little_groups_lines)