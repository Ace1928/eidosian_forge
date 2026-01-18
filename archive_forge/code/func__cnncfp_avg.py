from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
def _cnncfp_avg(data):
    """Global average pool"""
    return data.mean((1, 2))