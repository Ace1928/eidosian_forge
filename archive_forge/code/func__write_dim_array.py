import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _write_dim_array(self):
    if self.dimensions:
        self.fp.write(NC_DIMENSION)
        self._pack_int(len(self.dimensions))
        for name in self._dims:
            self._pack_string(name)
            length = self.dimensions[name]
            self._pack_int(length or 0)
    else:
        self.fp.write(ABSENT)