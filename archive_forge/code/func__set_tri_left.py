import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_tri_left(self):
    self._set_tri_down()
    self._transform = self._transform.rotate_deg(270)