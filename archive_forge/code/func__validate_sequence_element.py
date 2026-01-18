import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def _validate_sequence_element(t):
    return isinstance(t, Sequence) and len(t) == 2 and hasattr(t[0], '__array_interface__') and isinstance(t[1], Integral)