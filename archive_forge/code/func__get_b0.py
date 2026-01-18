from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def _get_b0(b0, units=None):
    if units is not None and b0 is integer_one:
        return b0 * units.molal
    else:
        return b0