import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_custom_marker(self, path):
    rescale = np.max(np.abs(path.vertices))
    self._transform = Affine2D().scale(0.5 / rescale)
    self._path = path