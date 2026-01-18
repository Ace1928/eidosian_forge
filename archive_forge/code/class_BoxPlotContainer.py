from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
class BoxPlotContainer:

    def __init__(self, artist_dict):
        self.boxes = artist_dict['boxes']
        self.medians = artist_dict['medians']
        self.whiskers = artist_dict['whiskers']
        self.caps = artist_dict['caps']
        self.fliers = artist_dict['fliers']
        self.means = artist_dict['means']
        self._label = None
        self._children = [*self.boxes, *self.medians, *self.whiskers, *self.caps, *self.fliers, *self.means]

    def __repr__(self):
        return f'<BoxPlotContainer object with {len(self.boxes)} boxes>'

    def __getitem__(self, idx):
        pair_slice = slice(2 * idx, 2 * idx + 2)
        return BoxPlotArtists(self.boxes[idx] if self.boxes else [], self.medians[idx] if self.medians else [], self.whiskers[pair_slice] if self.whiskers else [], self.caps[pair_slice] if self.caps else [], self.fliers[idx] if self.fliers else [], self.means[idx] if self.means else [])

    def __iter__(self):
        yield from (self[i] for i in range(len(self.boxes)))

    def get_label(self):
        return self._label

    def set_label(self, value):
        self._label = value

    def get_children(self):
        return self._children

    def remove(self):
        for child in self._children:
            child.remove()