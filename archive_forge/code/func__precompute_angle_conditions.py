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
def _precompute_angle_conditions(self, ivoronoi, voronoi):
    angle_conditions = []
    for iap, ap_dict in enumerate(self.neighbors_normalized_angles[ivoronoi]):
        angle_conditions.append([])
        ap = ap_dict['max']
        for _, vals in voronoi:
            angle_conditions[iap].append(vals['normalized_angle'] >= ap or np.isclose(vals['normalized_angle'], ap, rtol=0.0, atol=self.normalized_angle_tolerance / 2.0))
    return angle_conditions