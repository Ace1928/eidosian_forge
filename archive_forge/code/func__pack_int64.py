import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _pack_int64(self, value):
    self.fp.write(array(value, '>q').tobytes())