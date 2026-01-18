import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _read_dim_array(self):
    header = self.fp.read(4)
    if header not in [ZERO, NC_DIMENSION]:
        raise ValueError('Unexpected header.')
    count = self._unpack_int()
    for dim in range(count):
        name = self._unpack_string().decode('latin1')
        length = self._unpack_int() or None
        self.dimensions[name] = length
        self._dims.append(name)