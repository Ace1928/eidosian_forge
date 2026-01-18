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
class PlotSize(LinkedStream):
    """
    Returns the dimensions of a plot once it has been displayed.
    """
    width = param.Integer(default=None, constant=True, doc='The width of the plot in pixels')
    height = param.Integer(default=None, constant=True, doc='The height of the plot in pixels')
    scale = param.Number(default=1.0, constant=True, doc='\n       Scale factor to scale width and height values reported by the stream')

    def transform(self):
        return {'width': int(self.width * self.scale) if self.width else None, 'height': int(self.height * self.scale) if self.height else None}