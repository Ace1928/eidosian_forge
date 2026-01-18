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
class PlotReset(LinkedStream):
    """
    A stream signalling when a plot reset event has been triggered.
    """
    resetting = param.Boolean(default=False, constant=True, doc='\n        Whether a reset event is being signalled.')

    def __init__(self, *args, **params):
        super().__init__(self, *args, **dict(params, transient=True))