from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
def _color_to_cmap(color):
    """
    Create a light to dark cmap list from a base color
    """
    from .plotting.util import linear_gradient
    start_color = linear_gradient('#ffffff', color, 7)[2]
    end_color = linear_gradient(color, '#000000', 7)[2]
    return linear_gradient(start_color, end_color, 64)