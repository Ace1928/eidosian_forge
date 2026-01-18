import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_caretdown(self):
    self._transform = Affine2D().scale(0.5)
    self._snap_threshold = 3.0
    self._filled = False
    self._path = self._caret_path
    self._joinstyle = self._user_joinstyle or JoinStyle.miter