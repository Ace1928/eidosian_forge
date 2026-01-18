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
def _set_stream_parameters(self, **kwargs):
    """
        Sets the stream parameters which are expected to be declared
        constant.
        """
    with util.disable_constant(self):
        self.param.update(**kwargs)