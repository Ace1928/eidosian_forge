from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _get_animated_artists(self):
    """
        Convenience method to get all animated artists of the figure containing
        this widget, excluding those already present in self.artists.
        The returned tuple is not sorted by 'z_order': z_order sorting is
        valid only when considering all artists and not only a subset of all
        artists.
        """
    return tuple((a for ax_ in self.ax.get_figure().get_axes() for a in ax_.get_children() if a.get_animated() and a not in self.artists))