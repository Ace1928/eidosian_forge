from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
def _cnncfp_pad(data):
    """Pad the input"""
    pad_data = np.zeros((11, 113))
    return np.vstack([pad_data, data, pad_data])