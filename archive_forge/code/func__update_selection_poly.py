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
def _update_selection_poly(self, vmin, vmax):
    """
        Update the vertices of the *self.poly* slider in-place
        to cover the data range *vmin*, *vmax*.
        """
    verts = self.poly.xy
    if self.orientation == 'vertical':
        verts[0] = verts[4] = (0.25, vmin)
        verts[1] = (0.25, vmax)
        verts[2] = (0.75, vmax)
        verts[3] = (0.75, vmin)
    else:
        verts[0] = verts[4] = (vmin, 0.25)
        verts[1] = (vmin, 0.75)
        verts[2] = (vmax, 0.75)
        verts[3] = (vmax, 0.25)