from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def _cg_csm_separation_plane(self, coordination_geometry, sep_plane, local_plane, plane_separations, dist_tolerances=None, testing=False, tested_permutations=False, points_perfect=None):
    argref_separation = sep_plane.argsorted_ref_separation_perm
    plane_found = False
    permutations = []
    permutations_symmetry_measures = []
    if testing:
        separation_permutations = []
    dist_tolerances = dist_tolerances or DIST_TOLERANCES
    for dist_tolerance in dist_tolerances:
        algo = 'NOT_FOUND'
        separation = local_plane.indices_separate(self.local_geometry._coords, dist_tolerance)
        separation = sort_separation(separation)
        if separation_in_list(separation, plane_separations):
            continue
        if len(separation[1]) != len(sep_plane.plane_points):
            continue
        if len(separation[0]) == len(sep_plane.point_groups[0]):
            this_separation = separation
            plane_separations.append(this_separation)
        elif len(separation[0]) == len(sep_plane.point_groups[1]):
            this_separation = [list(separation[2]), list(separation[1]), list(separation[0])]
            plane_separations.append(this_separation)
        else:
            continue
        if sep_plane.ordered_plane:
            inp = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[1]]
            if sep_plane.ordered_point_groups[0]:
                pp_s0 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[0]]
                ordind_s0 = local_plane.project_and_to2dim_ordered_indices(pp_s0)
                sep0 = [this_separation[0][ii] for ii in ordind_s0]
            else:
                sep0 = list(this_separation[0])
            if sep_plane.ordered_point_groups[1]:
                pp_s2 = [pp for ip, pp in enumerate(self.local_geometry._coords) if ip in this_separation[2]]
                ordind_s2 = local_plane.project_and_to2dim_ordered_indices(pp_s2)
                sep2 = [this_separation[2][ii] for ii in ordind_s2]
            else:
                sep2 = list(this_separation[2])
            separation_perm = list(sep0)
            ordind = local_plane.project_and_to2dim_ordered_indices(inp)
            separation_perm.extend([this_separation[1][ii] for ii in ordind])
            algo = 'SEPARATION_PLANE_2POINTS_ORDERED'
            separation_perm.extend(sep2)
        else:
            separation_perm = list(this_separation[0])
            separation_perm.extend(this_separation[1])
            algo = 'SEPARATION_PLANE_2POINTS'
            separation_perm.extend(this_separation[2])
        if self.plane_safe_permutations:
            sep_perms = sep_plane.safe_separation_permutations(ordered_plane=sep_plane.ordered_plane, ordered_point_groups=sep_plane.ordered_point_groups)
        else:
            sep_perms = sep_plane.permutations
        for sep_perm in sep_perms:
            perm1 = [separation_perm[ii] for ii in sep_perm]
            pp = [perm1[ii] for ii in argref_separation]
            if isinstance(tested_permutations, set) and coordination_geometry.equivalent_indices is not None:
                tuple_ref_perm = coordination_geometry.ref_permutation(pp)
                if tuple_ref_perm in tested_permutations:
                    continue
                tested_permutations.add(tuple_ref_perm)
            permutations.append(pp)
            if testing:
                separation_permutations.append(sep_perm)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=pp)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures.append(sm_info)
        if plane_found:
            break
    if len(permutations_symmetry_measures) > 0:
        if testing:
            return (permutations_symmetry_measures, permutations, algo, separation_permutations)
        return (permutations_symmetry_measures, permutations, [sep_plane.algorithm_type] * len(permutations))
    if plane_found:
        if testing:
            return (permutations_symmetry_measures, permutations, [], [])
        return (permutations_symmetry_measures, permutations, [])
    if testing:
        return (None, None, None, None)
    return (None, None, None)