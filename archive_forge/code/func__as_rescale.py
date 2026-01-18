import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def _as_rescale(self, get, targetbitdepth):
    """Helper used by :meth:`asRGB8` and :meth:`asRGBA8`."""
    width, height, pixels, info = get()
    maxval = 2 ** info['bitdepth'] - 1
    targetmaxval = 2 ** targetbitdepth - 1
    factor = float(targetmaxval) / float(maxval)
    info['bitdepth'] = targetbitdepth

    def iterscale():
        for row in pixels:
            yield [int(round(x * factor)) for x in row]
    if maxval == targetmaxval:
        return (width, height, pixels, info)
    else:
        return (width, height, iterscale(), info)