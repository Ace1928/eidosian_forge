import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce

        Applies the given missing value to the data array.

        Returns a numpy.ma array, with any value equal to missing_value masked
        out (unless missing_value is None, in which case the original array is
        returned).
        