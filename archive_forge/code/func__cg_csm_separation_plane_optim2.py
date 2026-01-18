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
def _cg_csm_separation_plane_optim2(self, coordination_geometry, sepplane, local_plane, points_perfect=None, separation_indices=None):
    argref_separation = sepplane.argsorted_ref_separation_perm
    permutations = []
    permutations_symmetry_measures = []
    stop_search = False
    if sepplane.ordered_plane:
        inp = self.local_geometry.coords.take(separation_indices[1], axis=0)
        if sepplane.ordered_point_groups[0]:
            pp_s0 = self.local_geometry.coords.take(separation_indices[0], axis=0)
            ordind_s0 = local_plane.project_and_to2dim_ordered_indices(pp_s0)
            sep0 = separation_indices[0].take(ordind_s0)
        else:
            sep0 = separation_indices[0]
        if sepplane.ordered_point_groups[1]:
            pp_s2 = self.local_geometry.coords.take(separation_indices[2], axis=0)
            ordind_s2 = local_plane.project_and_to2dim_ordered_indices(pp_s2)
            sep2 = separation_indices[2].take(ordind_s2)
        else:
            sep2 = separation_indices[2]
        ordind = local_plane.project_and_to2dim_ordered_indices(inp)
        inp1 = separation_indices[1].take(ordind)
        separation_perm = np.concatenate((sep0, inp1, sep2))
    else:
        separation_perm = np.concatenate(separation_indices)
    if self.plane_safe_permutations:
        sep_perms = sepplane.safe_separation_permutations(ordered_plane=sepplane.ordered_plane, ordered_point_groups=sepplane.ordered_point_groups)
    else:
        sep_perms = sepplane.permutations
    for sep_perm in sep_perms:
        perm1 = separation_perm.take(sep_perm)
        pp = perm1.take(argref_separation)
        permutations.append(pp)
        points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=pp)
        sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
        sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
        permutations_symmetry_measures.append(sm_info)
    if len(permutations_symmetry_measures) > 0:
        return (permutations_symmetry_measures, permutations, [sepplane.algorithm_type] * len(permutations), stop_search)
    return ([], [], [], stop_search)