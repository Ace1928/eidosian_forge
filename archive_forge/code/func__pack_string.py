import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _pack_string(self, s):
    count = len(s)
    self._pack_int(count)
    self.fp.write(s.encode('latin1'))
    self.fp.write(b'\x00' * (-count % 4))