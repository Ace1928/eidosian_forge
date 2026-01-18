import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
def _theta_stretch(self):

    def theta_stretch(theta, scale):
        theta = np.deg2rad(theta)
        x = np.cos(theta)
        y = np.sin(theta)
        stheta = np.rad2deg(np.arctan2(scale * y, x))
        return (stheta + 360) % 360
    width = self.convert_xunits(self.width)
    height = self.convert_yunits(self.height)
    if width != height and (not (self.theta1 != self.theta2 and self.theta1 % 360 == self.theta2 % 360)):
        theta1 = theta_stretch(self.theta1, width / height)
        theta2 = theta_stretch(self.theta2, width / height)
        return (theta1, theta2, width, height)
    return (self.theta1, self.theta2, width, height)