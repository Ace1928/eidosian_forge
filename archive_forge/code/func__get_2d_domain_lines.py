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
def _get_2d_domain_lines(draw_domains) -> list[Scatter]:
    """
        Returns a list of Scatter objects tracing the domain lines on a
        2-dimensional chemical potential diagram.
        """
    x, y = ([], [])
    for pts in draw_domains.values():
        x.extend([*pts[:, 0].tolist(), None])
        y.extend([*pts[:, 1].tolist(), None])
    return [Scatter(x=x, y=y, mode='lines+markers', line={'color': 'black', 'width': 3}, showlegend=False)]