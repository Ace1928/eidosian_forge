from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def _create_plotly_ternary_support_lines(self):
    """
        Creates support lines which aid in seeing the ternary hull in three
        dimensions.

        Returns:
            go.Scatter3d plot of support lines for ternary phase diagram.
        """
    stable_entry_coords = dict(map(reversed, self.pd_plot_data[1].items()))
    elem_coords = [stable_entry_coords[entry] for entry in self._pd.el_refs.values()]
    x, y, z = ([], [], [])
    for line in itertools.combinations(elem_coords, 2):
        x.extend([line[0][0], line[1][0], None] * 2)
        y.extend([line[0][1], line[1][1], None] * 2)
        z.extend([0, 0, None, self._min_energy, self._min_energy, None])
    for elem in elem_coords:
        x.extend([elem[0], elem[0], None])
        y.extend([elem[1], elem[1], None])
        z.extend([0, self._min_energy, None])
    return go.Scatter3d(x=list(y), y=list(x), z=list(z), mode='lines', hoverinfo='none', line={'color': 'rgba (0, 0, 0, 0.4)', 'dash': 'solid', 'width': 1.0}, showlegend=False)