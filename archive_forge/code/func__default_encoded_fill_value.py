import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _default_encoded_fill_value(self):
    """
        The default encoded fill-value for this Variable's data type.
        """
    nc_type = REVERSE[self.typecode(), self.itemsize()]
    return FILLMAP[nc_type]