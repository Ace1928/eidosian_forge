import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_pentagon(self):
    self._transform = Affine2D().scale(0.5)
    self._snap_threshold = 5.0
    polypath = Path.unit_regular_polygon(5)
    if not self._half_fill():
        self._path = polypath
    else:
        verts = polypath.vertices
        y = (1 + np.sqrt(5)) / 4.0
        top = Path(verts[[0, 1, 4, 0]])
        bottom = Path(verts[[1, 2, 3, 4, 1]])
        left = Path([verts[0], verts[1], verts[2], [0, -y], verts[0]])
        right = Path([verts[0], verts[4], verts[3], [0, -y], verts[0]])
        self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
        self._alt_transform = self._transform
    self._joinstyle = self._user_joinstyle or JoinStyle.miter