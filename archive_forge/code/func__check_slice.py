from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
def _check_slice(s: slice) -> None:
    if s.start is not None and s.stop is not None and (s.start > s.stop):
        raise ValueError('Patch slices must have start < end, got %s' % s)
    if s.start is not None and s.start < 0 or (s.stop is not None and s.stop < 0) or (s.step is not None and s.step < 0):
        raise ValueError('Patch slices must have non-negative (start, stop, step) values, got %s' % s)