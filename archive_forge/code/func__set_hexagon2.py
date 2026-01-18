import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_hexagon2(self):
    self._transform = Affine2D().scale(0.5).rotate_deg(30)
    self._snap_threshold = None
    polypath = Path.unit_regular_polygon(6)
    if not self._half_fill():
        self._path = polypath
    else:
        verts = polypath.vertices
        x, y = (np.sqrt(3) / 4, 3 / 4.0)
        top = Path(verts[[1, 0, 5, 4, 1]])
        bottom = Path(verts[1:5])
        left = Path(np.concatenate([[(x, y)], verts[:3], [(-x, -y), (x, y)]]))
        right = Path(np.concatenate([[(x, y)], verts[5:2:-1], [(-x, -y)]]))
        self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
        self._alt_transform = self._transform
    self._joinstyle = self._user_joinstyle or JoinStyle.miter