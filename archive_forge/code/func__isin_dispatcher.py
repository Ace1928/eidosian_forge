import functools
import numpy as np
from numpy.core import overrides
def _isin_dispatcher(element, test_elements, assume_unique=None, invert=None, *, kind=None):
    return (element, test_elements)