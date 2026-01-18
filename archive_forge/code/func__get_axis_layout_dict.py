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
def _get_axis_layout_dict(elements: list[Element]) -> dict[str, str]:
    """Returns a Plotly layout dict for either 2-d or 3-d axes."""
    axes = ['xaxis', 'yaxis']
    layout_name = 'default_2d_axis_layout'
    if len(elements) == 3:
        axes.append('zaxis')
        layout_name = 'default_3d_axis_layout'

    def get_chempot_axis_title(element) -> str:
        return f'<br> μ<sub>{element}</sub> - μ<sub>{element}</sub><sup>o</sup> (eV)'
    axes_layout = {}
    for ax, el in zip(axes, elements):
        layout = plotly_layouts[layout_name].copy()
        layout['title'] = get_chempot_axis_title(el)
        axes_layout[ax] = layout
    return axes_layout