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
def _scale_polygon(self, event):
    """
        Scale the polygon selector points when the bounding box is moved or
        scaled.

        This is set as a callback on the bounding box RectangleSelector.
        """
    if not self._selection_completed:
        return
    if self._old_box_extents == self._box.extents:
        return
    x1, y1, w1, h1 = self._box._rect_bbox
    old_bbox = self._get_bbox()
    t = transforms.Affine2D().translate(-old_bbox.x0, -old_bbox.y0).scale(1 / old_bbox.width, 1 / old_bbox.height).scale(w1, h1).translate(x1, y1)
    new_verts = [(x, y) for x, y in t.transform(np.array(self.verts))]
    self._xys = [*new_verts, new_verts[0]]
    self._draw_polygon()
    self._old_box_extents = self._box.extents