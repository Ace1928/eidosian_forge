import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _footprint_is_sequence(footprint):
    if hasattr(footprint, '__array_interface__'):
        return False

    def _validate_sequence_element(t):
        return isinstance(t, Sequence) and len(t) == 2 and hasattr(t[0], '__array_interface__') and isinstance(t[1], Integral)
    if isinstance(footprint, Sequence):
        if not all((_validate_sequence_element(t) for t in footprint)):
            raise ValueError('All elements of footprint sequence must be a 2-tuple where the first element of the tuple is an ndarray and the second is an integer indicating the number of iterations.')
    else:
        raise ValueError('footprint must be either an ndarray or Sequence')
    return True