import collections
import gzip
import io
import logging
import struct
import numpy as np
def _fromfile(fin, dtype, count):
    """Reimplementation of numpy.fromfile."""
    return np.fromiter(_batched_generator(fin, count), dtype=dtype)