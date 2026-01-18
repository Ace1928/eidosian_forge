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
@_register_style(_style_list, name=']-')
class BracketA(_Curve):
    """An arrow with an outward square bracket at its start."""
    arrow = ']-'

    def __init__(self, widthA=1.0, lengthA=0.2, angleA=0):
        """
            Parameters
            ----------
            widthA : float, default: 1.0
                Width of the bracket.
            lengthA : float, default: 0.2
                Length of the bracket.
            angleA : float, default: 0 degrees
                Orientation of the bracket, as a counterclockwise angle.
                0 degrees means perpendicular to the line.
            """
        super().__init__(widthA=widthA, lengthA=lengthA, angleA=angleA)