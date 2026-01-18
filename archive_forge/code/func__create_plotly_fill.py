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
def _create_plotly_fill(self):
    """
        Creates shaded mesh traces for coloring the hull.

        For tenrary_3d plots, the color shading is based on formation energy.

        Returns:
            go.Mesh3d plot
        """
    traces = []
    pd = self._pd
    if self._dim == 3 and self.ternary_style == '2d':
        fillcolors = itertools.cycle(plotly_layouts['default_fill_colors'])
        el_a, el_b, el_c = pd.elements
        for _idx, facet in enumerate(pd.facets):
            a = []
            b = []
            c = []
            e0, e1, e2 = sorted((pd.qhull_entries[facet[idx]] for idx in range(3)), key=lambda x: x.reduced_formula)
            a = [e0.composition[el_a], e1.composition[el_a], e2.composition[el_a]]
            b = [e0.composition[el_b], e1.composition[el_b], e2.composition[el_b]]
            c = [e0.composition[el_c], e1.composition[el_c], e2.composition[el_c]]
            e_strs = []
            for entry in (e0, e1, e2):
                if hasattr(entry, 'original_entry'):
                    entry = entry.original_entry
                e_strs.append(htmlify(entry.reduced_formula))
            name = f'{e_strs[0]}—{e_strs[1]}—{e_strs[2]}'
            traces += [go.Scatterternary(a=a, b=b, c=c, mode='lines', fill='toself', line={'width': 0}, fillcolor=next(fillcolors), opacity=0.15, hovertemplate='<extra></extra>', name=name, showlegend=False)]
    elif self._dim == 3 and self.ternary_style == '3d':
        facets = np.array(self._pd.facets)
        coords = np.array([triangular_coord(c) for c in zip(self._pd.qhull_data[:-1, 0], self._pd.qhull_data[:-1, 1])])
        energies = np.array([self._pd.get_form_energy_per_atom(entry) for entry in self._pd.qhull_entries])
        traces.append(go.Mesh3d(x=list(coords[:, 1]), y=list(coords[:, 0]), z=list(energies), i=list(facets[:, 1]), j=list(facets[:, 0]), k=list(facets[:, 2]), opacity=0.7, intensity=list(energies), colorscale=plotly_layouts['stable_colorscale'], colorbar={'title': 'Formation energy<br>(eV/atom)', 'x': 0.9, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}, hoverinfo='none', lighting={'diffuse': 0.0, 'ambient': 1.0}, name='Convex Hull (shading)', flatshading=True, showlegend=True))
    elif self._dim == 4:
        all_data = np.array(pd.qhull_data)
        fillcolors = itertools.cycle(plotly_layouts['default_fill_colors'])
        for _idx, facet in enumerate(pd.facets):
            xs, ys, zs = ([], [], [])
            for v in facet:
                x, y, z = tet_coord(all_data[v, 0:3])
                xs.append(x)
                ys.append(y)
                zs.append(z)
            traces += [go.Mesh3d(x=xs, y=ys, z=zs, opacity=0.05, alphahull=-1, flatshading=True, hoverinfo='skip', color=next(fillcolors))]
    return traces