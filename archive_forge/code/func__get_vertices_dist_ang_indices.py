from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
@staticmethod
def _get_vertices_dist_ang_indices(parameter_indices_list):
    pp0 = [pp[0] for pp in parameter_indices_list]
    pp1 = [pp[1] for pp in parameter_indices_list]
    min_idist = min(pp0)
    min_iang = min(pp1)
    max_idist = max(pp0)
    max_iang = max(pp1)
    i_min_angs = np.argwhere(np.array(pp1) == min_iang)
    i_max_dists = np.argwhere(np.array(pp0) == max_idist)
    pp0_at_min_iang = [pp0[ii[0]] for ii in i_min_angs]
    pp1_at_max_idist = [pp1[ii[0]] for ii in i_max_dists]
    max_idist_at_min_iang = max(pp0_at_min_iang)
    min_iang_at_max_idist = min(pp1_at_max_idist)
    p1 = (min_idist, min_iang)
    p2 = (max_idist_at_min_iang, min_iang)
    p3 = (max_idist_at_min_iang, min_iang_at_max_idist)
    p4 = (max_idist, min_iang_at_max_idist)
    p5 = (max_idist, max_iang)
    p6 = (min_idist, max_iang)
    return [p1, p2, p3, p4, p5, p6]