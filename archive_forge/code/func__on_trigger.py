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
def _on_trigger(self):
    if any((e.type == 'triggered' for e in self._events)):
        self._memoize_counter += 1