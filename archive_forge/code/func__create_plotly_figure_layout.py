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
def _create_plotly_figure_layout(self, label_stable=True):
    """
        Creates layout for plotly phase diagram figure and updates with
        figure annotations.

        Args:
            label_stable (bool): Whether to label stable compounds

        Returns:
            Dictionary with Plotly figure layout settings.
        """
    annotations_list = None
    layout = {}
    if label_stable:
        annotations_list = self._create_plotly_element_annotations()
    if self._dim == 1:
        layout = plotly_layouts['default_unary_layout'].copy()
    if self._dim == 2:
        layout = plotly_layouts['default_binary_layout'].copy()
        layout['xaxis']['title'] = f'Composition (Fraction {self._pd.elements[1]})'
        layout['annotations'] = annotations_list
    elif self._dim == 3 and self.ternary_style == '2d':
        layout = plotly_layouts['default_ternary_2d_layout'].copy()
        for el, axis in zip(self._pd.elements, ['a', 'b', 'c']):
            el_ref = self._pd.el_refs[el]
            clean_formula = str(el_ref.elements[0])
            if hasattr(el_ref, 'original_entry'):
                clean_formula = htmlify(el_ref.original_entry.reduced_formula)
            layout['ternary'][axis + 'axis']['title'] = {'text': clean_formula, 'font': {'size': 24}}
    elif self._dim == 3 and self.ternary_style == '3d':
        layout = plotly_layouts['default_ternary_3d_layout'].copy()
        layout['scene']['annotations'] = annotations_list
    elif self._dim == 4:
        layout = plotly_layouts['default_quaternary_layout'].copy()
        layout['scene']['annotations'] = annotations_list
    return layout