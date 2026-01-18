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
def _get_border_hyperplanes(self) -> np.ndarray:
    """Returns an array of the bounding hyperplanes given by elemental limits."""
    border_hyperplanes = np.array([[0] * (self.dim + 1)] * (2 * self.dim))
    for idx, limit in enumerate(self.lims):
        border_hyperplanes[2 * idx, idx] = -1
        border_hyperplanes[2 * idx, -1] = limit[0]
        border_hyperplanes[2 * idx + 1, idx] = 1
        border_hyperplanes[2 * idx + 1, -1] = limit[1]
    return border_hyperplanes