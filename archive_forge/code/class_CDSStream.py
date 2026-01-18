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
class CDSStream(LinkedStream):
    """
    A Stream that syncs a bokeh ColumnDataSource with python.
    """
    data = param.Dict(constant=True, doc='\n        Data synced from Bokeh ColumnDataSource supplied as a\n        dictionary of columns, where each column is a list of values\n        (for point-like data) or list of lists of values (for\n        path-like data).')