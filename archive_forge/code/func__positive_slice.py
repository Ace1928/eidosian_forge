import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def _positive_slice(slicer):
    """Return full slice `slicer` enforcing positive step size

    `slicer` assumed full in the sense of :func:`fill_slicer`
    """
    start, stop, step = (slicer.start, slicer.stop, slicer.step)
    if step > 0:
        return slicer
    if stop is None:
        stop = -1
    gap = stop - start
    n = gap / step
    n = int(n) - 1 if int(n) == n else int(n)
    end = start + n * step
    return slice(end, start + 1, -step)