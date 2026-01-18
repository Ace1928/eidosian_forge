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
def _create_plotly_stable_labels(self, label_stable=True):
    """
        Creates a (hidable) scatter trace containing labels of stable phases.
        Contains some functionality for creating sensible label positions. This method
        does not apply to 2D ternary plots (stable labels are turned off).

        Returns:
            go.Scatter (or go.Scatter3d) plot
        """
    x, y, z, text, textpositions = ([], [], [], [], [])
    stable_labels_plot = min_energy_x = None
    offset_2d = 0.008
    offset_3d = 0.01
    energy_offset = -0.05 * self._min_energy
    if self._dim == 2:
        min_energy_x = min(list(self.pd_plot_data[1]), key=lambda c: c[1])[0]
    for coords, entry in self.pd_plot_data[1].items():
        if entry.composition.is_element:
            continue
        x_coord = coords[0]
        y_coord = coords[1]
        textposition = None
        if self._dim == 2:
            textposition = 'bottom left'
            if x_coord >= min_energy_x:
                textposition = 'bottom right'
                x_coord += offset_2d
            else:
                x_coord -= offset_2d
            y_coord -= offset_2d + 0.005
        elif self._dim == 3 and self.ternary_style == '3d':
            textposition = 'middle center'
            if coords[0] > 0.5:
                x_coord += offset_3d
            else:
                x_coord -= offset_3d
            if coords[1] > 0.866 / 2:
                y_coord -= offset_3d
            else:
                y_coord += offset_3d
            z.append(self._pd.get_form_energy_per_atom(entry) + energy_offset)
        elif self._dim == 4:
            x_coord = x_coord - offset_3d
            y_coord = y_coord - offset_3d
            textposition = 'bottom right'
            z.append(coords[2])
        x.append(x_coord)
        y.append(y_coord)
        textpositions.append(textposition)
        comp = entry.composition
        if hasattr(entry, 'original_entry'):
            comp = entry.original_entry.composition
        formula = comp.reduced_formula
        text.append(htmlify(formula))
    visible = True
    if not label_stable or self._dim == 4:
        visible = 'legendonly'
    plot_args = {'text': text, 'textposition': textpositions, 'mode': 'text', 'name': 'Labels (stable)', 'hoverinfo': 'skip', 'opacity': 1.0, 'visible': visible, 'showlegend': True}
    if self._dim == 2:
        stable_labels_plot = go.Scatter(x=x, y=y, **plot_args)
    elif self._dim == 3 and self.ternary_style == '3d':
        stable_labels_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
    elif self._dim == 4:
        stable_labels_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)
    return stable_labels_plot