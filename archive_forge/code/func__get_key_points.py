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
def _get_key_points(self):
    decimals = ceil(-1 * np.log10(self._atol)) - 1
    bz = self._rec_lattice.get_wigner_seitz_cell()
    key_points = []
    face_center_inds = []
    bz_as_key_point_inds = []
    for idx, facet in enumerate(bz):
        for j, vert in enumerate(facet):
            vert = self._rec_lattice.get_fractional_coords(vert)
            bz[idx][j] = vert
    pop = []
    for idx, facet in enumerate(bz):
        rounded_facet = np.around(facet, decimals=decimals)
        u, indices = np.unique(rounded_facet, axis=0, return_index=True)
        if len(u) in [1, 2]:
            pop.append(idx)
        else:
            bz[idx] = [facet[j] for j in np.sort(indices)]
    bz = [bz[i] for i in range(len(bz)) if i not in pop]
    for idx, facet in enumerate(bz):
        bz_as_key_point_inds.append([])
        for j, vert in enumerate(facet):
            edge_center = (vert + facet[j + 1]) / 2 if j != len(facet) - 1 else (vert + facet[0]) / 2.0
            duplicatevert = False
            duplicateedge = False
            for k, point in enumerate(key_points):
                if np.allclose(vert, point, atol=self._atol):
                    bz_as_key_point_inds[idx].append(k)
                    duplicatevert = True
                    break
            for k, point in enumerate(key_points):
                if np.allclose(edge_center, point, atol=self._atol):
                    bz_as_key_point_inds[idx].append(k)
                    duplicateedge = True
                    break
            if not duplicatevert:
                key_points.append(vert)
                bz_as_key_point_inds[idx].append(len(key_points) - 1)
            if not duplicateedge:
                key_points.append(edge_center)
                bz_as_key_point_inds[idx].append(len(key_points) - 1)
        if len(facet) == 4:
            face_center = (facet[0] + facet[1] + facet[2] + facet[3]) / 4.0
            key_points.append(face_center)
            face_center_inds.append(len(key_points) - 1)
            bz_as_key_point_inds[idx].append(len(key_points) - 1)
        else:
            face_center = (facet[0] + facet[1] + facet[2] + facet[3] + facet[4] + facet[5]) / 6.0
            key_points.append(face_center)
            face_center_inds.append(len(key_points) - 1)
            bz_as_key_point_inds[idx].append(len(key_points) - 1)
    key_points.append(np.array([0, 0, 0]))
    return (key_points, bz_as_key_point_inds, face_center_inds)