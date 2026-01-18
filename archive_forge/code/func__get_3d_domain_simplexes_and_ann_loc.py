from a list of entries within a chemical system containing 2 or more elements. The
from __future__ import annotations
import json
import os
import warnings
from functools import lru_cache
from itertools import groupby
from typing import TYPE_CHECKING
import numpy as np
import plotly.express as px
from monty.json import MSONable
from plotly.graph_objects import Figure, Mesh3d, Scatter, Scatter3d
from scipy.spatial import ConvexHull, HalfspaceIntersection
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition, Element
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.string import htmlify
@staticmethod
def _get_3d_domain_simplexes_and_ann_loc(points_3d: np.ndarray) -> tuple[list[Simplex], np.ndarray]:
    """
        Returns a list of Simplex objects and coordinates of annotation for one
        domain in a 3-d chemical potential diagram. Uses PCA to project domain
        into 2-dimensional space so that ConvexHull can be used to identify the
        bounding polygon.
        """
    points_2d, _v, w = simple_pca(points_3d, k=2)
    domain = ConvexHull(points_2d)
    centroid_2d = get_centroid_2d(points_2d[domain.vertices])
    ann_loc = centroid_2d @ w.T + np.mean(points_3d.T, axis=1)
    simplexes = [Simplex(points_3d[indices]) for indices in domain.simplices]
    return (simplexes, ann_loc)