import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
def _cells_unpack_weights(cells, args):
    for cell in cells:
        args = cell.unpack_weights(args)
    return args