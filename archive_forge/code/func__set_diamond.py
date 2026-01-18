import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_diamond(self):
    self._transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
    self._snap_threshold = 5.0
    if not self._half_fill():
        self._path = Path.unit_rectangle()
    else:
        self._path = Path([[0, 0], [1, 0], [1, 1], [0, 0]])
        self._alt_path = Path([[0, 0], [0, 1], [1, 1], [0, 0]])
        fs = self.get_fillstyle()
        rotate = {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs]
        self._transform.rotate_deg(rotate)
        self._alt_transform = self._transform
    self._joinstyle = self._user_joinstyle or JoinStyle.miter