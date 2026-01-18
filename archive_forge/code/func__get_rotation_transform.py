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
def _get_rotation_transform(self):
    aspect_ratio = self.ax._get_aspect_ratio()
    return Affine2D().translate(-self.center[0], -self.center[1]).scale(1, aspect_ratio).rotate(self._rotation).scale(1, 1 / aspect_ratio).translate(*self.center)