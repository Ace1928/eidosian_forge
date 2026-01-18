import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
def _cells_state_shape(cells):
    return sum([c.state_shape for c in cells], [])