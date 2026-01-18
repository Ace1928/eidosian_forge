import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _unpack_string(self):
    count = self._unpack_int()
    s = self.fp.read(count).rstrip(b'\x00')
    self.fp.read(-count % 4)
    return s