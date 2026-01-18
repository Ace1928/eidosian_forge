import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _read_att_array(self):
    header = self.fp.read(4)
    if header not in [ZERO, NC_ATTRIBUTE]:
        raise ValueError('Unexpected header.')
    count = self._unpack_int()
    attributes = {}
    for attr in range(count):
        name = self._unpack_string().decode('latin1')
        attributes[name] = self._read_att_values()
    return attributes