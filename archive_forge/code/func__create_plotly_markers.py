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
def _create_plotly_markers(self, highlight_entries=None, label_uncertainties=False):
    """
        Creates stable and unstable marker plots for overlaying on the phase diagram.

        Returns:
            Tuple of Plotly go.Scatter (unary, binary), go.Scatterternary(ternary_2d),
            or go.Scatter3d (ternary_3d, quaternary) objects in order:
            (stable markers, unstable markers)
        """

    def get_marker_props(coords, entries):
        """Method for getting marker locations, hovertext, and error bars
            from pd_plot_data.
            """
        x, y, z, texts, energies, uncertainties = ([], [], [], [], [], [])
        is_stable = [entry in self._pd.stable_entries for entry in entries]
        for coord, entry, stable in zip(coords, entries, is_stable):
            energy = round(self._pd.get_form_energy_per_atom(entry), 3)
            entry_id = getattr(entry, 'entry_id', 'no ID')
            comp = entry.composition
            if hasattr(entry, 'original_entry'):
                orig_entry = entry.original_entry
                comp = orig_entry.composition
                entry_id = getattr(orig_entry, 'entry_id', 'no ID')
            formula = comp.reduced_formula
            clean_formula = htmlify(formula)
            label = f'{clean_formula} ({entry_id}) <br> {energy} eV/atom'
            if not stable:
                e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
                if e_above_hull > self.show_unstable:
                    continue
                label += f' ({e_above_hull:+} eV/atom)'
                energies.append(e_above_hull)
            else:
                uncertainty = 0
                label += ' (Stable)'
                if hasattr(entry, 'correction_uncertainty_per_atom') and label_uncertainties:
                    uncertainty = round(entry.correction_uncertainty_per_atom, 4)
                    label += f'<br> (Error: +/- {uncertainty} eV/atom)'
                uncertainties.append(uncertainty)
                energies.append(energy)
            texts.append(label)
            if self._dim == 3 and self.ternary_style == '2d':
                for el, axis in zip(self._pd.elements, [x, y, z]):
                    axis.append(entry.composition[el])
            else:
                x.append(coord[0])
                y.append(coord[1])
                if self._dim == 3:
                    z.append(energy)
                elif self._dim == 4:
                    z.append(coord[2])
        return {'x': x, 'y': y, 'z': z, 'texts': texts, 'energies': energies, 'uncertainties': uncertainties}
    if highlight_entries is None:
        highlight_entries = []
    stable_coords, stable_entries = ([], [])
    unstable_coords, unstable_entries = ([], [])
    highlight_coords, highlight_ents = ([], [])
    for coord, entry in zip(self.pd_plot_data[1], self.pd_plot_data[1].values()):
        if entry in highlight_entries:
            highlight_coords.append(coord)
            highlight_ents.append(entry)
        else:
            stable_coords.append(coord)
            stable_entries.append(entry)
    for coord, entry in zip(self.pd_plot_data[2].values(), self.pd_plot_data[2]):
        if entry in highlight_entries:
            highlight_coords.append(coord)
            highlight_ents.append(entry)
        else:
            unstable_coords.append(coord)
            unstable_entries.append(entry)
    stable_props = get_marker_props(stable_coords, stable_entries)
    unstable_props = get_marker_props(unstable_coords, unstable_entries)
    highlight_props = get_marker_props(highlight_coords, highlight_entries)
    stable_markers, unstable_markers, highlight_markers = ({}, {}, {})
    if self._dim == 1:
        stable_markers = plotly_layouts['default_unary_marker_settings'].copy()
        unstable_markers = plotly_layouts['default_unary_marker_settings'].copy()
        stable_markers.update(x=[0] * len(stable_props['y']), y=list(stable_props['x']), name='Stable', marker={'color': 'darkgreen', 'size': 20, 'line': {'color': 'black', 'width': 2}, 'symbol': 'star'}, opacity=0.9, hovertext=stable_props['texts'], error_y={'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
        plotly_layouts['unstable_colorscale'].copy()
        unstable_markers.update(x=[0] * len(unstable_props['y']), y=list(unstable_props['x']), name='Above Hull', marker={'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 16, 'symbol': 'diamond-wide', 'line': {'color': 'black', 'width': 2}}, hovertext=unstable_props['texts'], opacity=0.9)
        if highlight_entries:
            highlight_markers = plotly_layouts['default_unary_marker_settings'].copy()
            highlight_markers.update({'x': [0] * len(highlight_props['y']), 'y': list(highlight_props['x']), 'name': 'Highlighted', 'marker': {'color': 'mediumvioletred', 'size': 22, 'line': {'color': 'black', 'width': 2}, 'symbol': 'square'}, 'opacity': 0.9, 'hovertext': highlight_props['texts'], 'error_y': {'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5}})
    if self._dim == 2:
        stable_markers = plotly_layouts['default_binary_marker_settings'].copy()
        unstable_markers = plotly_layouts['default_binary_marker_settings'].copy()
        stable_markers.update(x=list(stable_props['x']), y=list(stable_props['y']), name='Stable', marker={'color': 'darkgreen', 'size': 16, 'line': {'color': 'black', 'width': 2}}, opacity=0.99, hovertext=stable_props['texts'], error_y={'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
        unstable_markers.update({'x': list(unstable_props['x']), 'y': list(unstable_props['y']), 'name': 'Above Hull', 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 7, 'symbol': 'diamond', 'line': {'color': 'black', 'width': 1}, 'opacity': 0.8}, 'hovertext': unstable_props['texts']})
        if highlight_entries:
            highlight_markers = plotly_layouts['default_binary_marker_settings'].copy()
            highlight_markers.update(x=list(highlight_props['x']), y=list(highlight_props['y']), name='Highlighted', marker={'color': 'mediumvioletred', 'size': 16, 'line': {'color': 'black', 'width': 2}, 'symbol': 'square'}, opacity=0.99, hovertext=highlight_props['texts'], error_y={'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
    elif self._dim == 3 and self.ternary_style == '2d':
        stable_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
        unstable_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
        stable_markers.update({'a': list(stable_props['x']), 'b': list(stable_props['y']), 'c': list(stable_props['z']), 'name': 'Stable', 'hovertext': stable_props['texts'], 'marker': {'color': 'green', 'line': {'width': 2.0, 'color': 'black'}, 'symbol': 'circle', 'size': 15}})
        unstable_markers.update({'a': unstable_props['x'], 'b': unstable_props['y'], 'c': unstable_props['z'], 'name': 'Above Hull', 'hovertext': unstable_props['texts'], 'marker': {'color': unstable_props['energies'], 'opacity': 0.8, 'colorscale': plotly_layouts['unstable_colorscale'], 'line': {'width': 1, 'color': 'black'}, 'size': 7, 'symbol': 'diamond', 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}})
        if highlight_entries:
            highlight_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
            highlight_markers.update({'a': list(highlight_props['x']), 'b': list(highlight_props['y']), 'c': list(highlight_props['z']), 'name': 'Highlighted', 'hovertext': highlight_props['texts'], 'marker': {'color': 'mediumvioletred', 'line': {'width': 2.0, 'color': 'black'}, 'symbol': 'square', 'size': 16}})
    elif self._dim == 3 and self.ternary_style == '3d':
        stable_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
        unstable_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
        stable_markers.update({'x': list(stable_props['y']), 'y': list(stable_props['x']), 'z': list(stable_props['z']), 'name': 'Stable', 'marker': {'color': '#1e1e1f', 'size': 11, 'opacity': 0.99}, 'hovertext': stable_props['texts'], 'error_z': {'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'darkgray', 'width': 10, 'thickness': 5}})
        unstable_markers.update({'x': unstable_props['y'], 'y': unstable_props['x'], 'z': unstable_props['z'], 'name': 'Above Hull', 'hovertext': unstable_props['texts'], 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 5, 'line': {'color': 'black', 'width': 1}, 'symbol': 'diamond', 'opacity': 0.7, 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}})
        if highlight_entries:
            highlight_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
            highlight_markers.update({'x': list(highlight_props['y']), 'y': list(highlight_props['x']), 'z': list(highlight_props['z']), 'name': 'Highlighted', 'marker': {'size': 12, 'opacity': 0.99, 'symbol': 'square', 'color': 'mediumvioletred'}, 'hovertext': highlight_props['texts'], 'error_z': {'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'darkgray', 'width': 10, 'thickness': 5}})
    elif self._dim == 4:
        stable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
        unstable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
        stable_markers.update({'x': stable_props['x'], 'y': stable_props['y'], 'z': stable_props['z'], 'name': 'Stable', 'marker': {'size': 7, 'opacity': 0.99, 'color': 'darkgreen', 'line': {'color': 'black', 'width': 1}}, 'hovertext': stable_props['texts']})
        unstable_markers.update({'x': unstable_props['x'], 'y': unstable_props['y'], 'z': unstable_props['z'], 'name': 'Above Hull', 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 5, 'symbol': 'diamond', 'line': {'color': 'black', 'width': 1}, 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}, 'hovertext': unstable_props['texts'], 'visible': 'legendonly'})
        if highlight_entries:
            highlight_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
            highlight_markers.update({'x': highlight_props['x'], 'y': highlight_props['y'], 'z': highlight_props['z'], 'name': 'Highlighted', 'marker': {'size': 9, 'opacity': 0.99, 'symbol': 'square', 'color': 'mediumvioletred', 'line': {'color': 'black', 'width': 1}}, 'hovertext': highlight_props['texts']})
    highlight_marker_plot = None
    if self._dim in [1, 2]:
        stable_marker_plot, unstable_marker_plot = (go.Scatter(**markers) for markers in [stable_markers, unstable_markers])
        if highlight_entries:
            highlight_marker_plot = go.Scatter(**highlight_markers)
    elif self._dim == 3 and self.ternary_style == '2d':
        stable_marker_plot, unstable_marker_plot = (go.Scatterternary(**markers) for markers in [stable_markers, unstable_markers])
        if highlight_entries:
            highlight_marker_plot = go.Scatterternary(**highlight_markers)
    else:
        stable_marker_plot, unstable_marker_plot = (go.Scatter3d(**markers) for markers in [stable_markers, unstable_markers])
        if highlight_entries:
            highlight_marker_plot = go.Scatter3d(**highlight_markers)
    return (stable_marker_plot, unstable_marker_plot, highlight_marker_plot)