from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
def _cnncfp_superframes(data):
    """Segment input into superframes"""
    from ..utils import segment_axis
    return segment_axis(data, 3, 1, axis=0)