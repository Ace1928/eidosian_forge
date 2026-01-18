import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def _iter_break_from_left_to_right(self):
    for left, right in self._a._iter_break_from_left_to_right():
        yield (left, right + self._b)
    for left, right in self._b._iter_break_from_left_to_right():
        yield (self._a + left, right)