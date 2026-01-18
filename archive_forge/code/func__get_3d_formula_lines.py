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
def _get_3d_formula_lines(draw_domains: dict[str, np.ndarray], formula_colors: list[str] | None) -> list[Scatter3d]:
    """Returns a list of Scatter3d objects defining the bounding polyhedra."""
    if formula_colors is None:
        formula_colors = px.colors.qualitative.Dark2
    lines = []
    for idx, (formula, coords) in enumerate(draw_domains.items()):
        points_3d = coords[:, :3]
        domain = ConvexHull(points_3d[:, :-1])
        simplexes = [Simplex(points_3d[indices]) for indices in domain.simplices]
        x, y, z = ([], [], [])
        for s in simplexes:
            x.extend([*s.coords[:, 0].tolist(), None])
            y.extend([*s.coords[:, 1].tolist(), None])
            z.extend([*s.coords[:, 2].tolist(), None])
        line = Scatter3d(x=x, y=y, z=z, mode='lines', line={'width': 8, 'color': formula_colors[idx]}, opacity=1.0, name=f'{formula} (lines)')
        lines.append(line)
    return lines