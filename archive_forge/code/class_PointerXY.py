import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class PointerXY(LinkedStream):
    """
    A pointer position along the x- and y-axes in data coordinates which
    may numeric or categorical dimensions.

    With the appropriate plotting backend, this corresponds to the
    position of the mouse/trackpad pointer. If the pointer is outside
    the plot bounds, the position values are set to None.
    """
    x = param.ClassSelector(class_=pointer_types, default=None, constant=True, doc='\n           Pointer position along the x-axis in data coordinates')
    y = param.ClassSelector(class_=pointer_types, default=None, constant=True, doc='\n           Pointer position along the y-axis in data coordinates')