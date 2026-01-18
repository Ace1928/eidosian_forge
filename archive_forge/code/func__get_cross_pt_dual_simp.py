from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def _get_cross_pt_dual_simp(self, dual_simp):
    """
        |normal| = 1, e_surf is plane's distance to (0, 0, 0),
        plane function:
            normal[0]x + normal[1]y + normal[2]z = e_surf.

        from self:
            normal_e_m to get the plane functions
            dual_simp: (i, j, k) simplices from the dual convex hull
                i, j, k: plane index(same order in normal_e_m)
        """
    matrix_surfs = [self.facets[dual_simp[i]].normal for i in range(3)]
    matrix_e = [self.facets[dual_simp[i]].e_surf for i in range(3)]
    return np.dot(np.linalg.inv(matrix_surfs), matrix_e)