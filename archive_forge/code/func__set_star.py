import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_star(self):
    self._transform = Affine2D().scale(0.5)
    self._snap_threshold = 5.0
    polypath = Path.unit_regular_star(5, innerCircle=0.381966)
    if not self._half_fill():
        self._path = polypath
    else:
        verts = polypath.vertices
        top = Path(np.concatenate([verts[0:4], verts[7:10], verts[0:1]]))
        bottom = Path(np.concatenate([verts[3:8], verts[3:4]]))
        left = Path(np.concatenate([verts[0:6], verts[0:1]]))
        right = Path(np.concatenate([verts[0:1], verts[5:10], verts[0:1]]))
        self._path, self._alt_path = {'top': (top, bottom), 'bottom': (bottom, top), 'left': (left, right), 'right': (right, left)}[self.get_fillstyle()]
        self._alt_transform = self._transform
    self._joinstyle = self._user_joinstyle or JoinStyle.bevel