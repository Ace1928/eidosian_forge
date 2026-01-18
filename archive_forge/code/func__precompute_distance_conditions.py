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
def _precompute_distance_conditions(self, ivoronoi, voronoi):
    distance_conditions = []
    for idp, dp_dict in enumerate(self.neighbors_normalized_distances[ivoronoi]):
        distance_conditions.append([])
        dp = dp_dict['max']
        for _, vals in voronoi:
            distance_conditions[idp].append(vals['normalized_distance'] <= dp or np.isclose(vals['normalized_distance'], dp, rtol=0.0, atol=self.normalized_distance_tolerance / 2.0))
    return distance_conditions