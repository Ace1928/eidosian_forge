from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def _extract_cell_info(self, site_idx, sites, targets, voro, compute_adj_neighbors=False):
    """Get the information about a certain atom from the results of a tessellation.

        Args:
            site_idx (int) - Index of the atom in question
            sites ([Site]) - List of all sites in the tessellation
            targets ([Element]) - Target elements
            voro - Output of qvoronoi
            compute_adj_neighbors (boolean) - Whether to compute which neighbors are adjacent

        Returns:
            A dict of sites sharing a common Voronoi facet. Key is facet id
             (not useful) and values are dictionaries containing statistics
             about the facet:
                - site: Pymatgen site
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
                - adj_neighbors - Facet id's for the adjacent neighbors
        """
    all_vertices = voro.vertices
    center_coords = sites[site_idx].coords
    results = {}
    for nn, vind in voro.ridge_dict.items():
        if site_idx in nn:
            other_site = nn[0] if nn[1] == site_idx else nn[1]
            if -1 in vind:
                if self.allow_pathological:
                    continue
                raise RuntimeError('This structure is pathological, infinite vertex in the Voronoi construction')
            facets = [all_vertices[idx] for idx in vind]
            angle = solid_angle(center_coords, facets)
            volume = 0
            for j, k in zip(vind[1:], vind[2:]):
                volume += vol_tetra(center_coords, all_vertices[vind[0]], all_vertices[j], all_vertices[k])
            face_dist = np.linalg.norm(center_coords - sites[other_site].coords) / 2
            face_area = 3 * volume / face_dist
            normal = np.subtract(sites[other_site].coords, center_coords)
            normal /= np.linalg.norm(normal)
            results[other_site] = {'site': sites[other_site], 'normal': normal, 'solid_angle': angle, 'volume': volume, 'face_dist': face_dist, 'area': face_area, 'n_verts': len(vind)}
            if compute_adj_neighbors:
                results[other_site]['verts'] = vind
    if len(results) == 0:
        raise ValueError('No Voronoi neighbors found for site - try increasing cutoff')
    result_weighted = {}
    for nn_index, nn_stats in results.items():
        nn = nn_stats['site']
        if nn.is_ordered:
            if nn.specie in targets:
                result_weighted[nn_index] = nn_stats
        else:
            for disordered_sp in nn.species:
                if disordered_sp in targets:
                    result_weighted[nn_index] = nn_stats
    if compute_adj_neighbors:
        adj_neighbors = {idx: [] for idx in result_weighted}
        for a_ind, a_nn_info in result_weighted.items():
            a_verts = set(a_nn_info['verts'])
            for b_ind, b_nninfo in result_weighted.items():
                if b_ind > a_ind:
                    continue
                if len(a_verts.intersection(b_nninfo['verts'])) == 2:
                    adj_neighbors[a_ind].append(b_ind)
                    adj_neighbors[b_ind].append(a_ind)
        for key, neighbors in adj_neighbors.items():
            result_weighted[key]['adj_neighbors'] = neighbors
    return result_weighted