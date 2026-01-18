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
def _register_streams(self, streams):
    """
        Register callbacks to watch for changes to input streams
        """
    for stream in streams:
        self._register_stream(stream)