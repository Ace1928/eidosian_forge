import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _get_missing_value(self):
    """
        Returns the value denoting "no data" for this variable.

        If this variable does not have a missing/fill value, returns None.

        If both _FillValue and missing_value are given, give precedence to
        _FillValue. The netCDF standard gives special meaning to _FillValue;
        missing_value is  just used for compatibility with old datasets.
        """
    if '_FillValue' in self._attributes:
        missing_value = self._attributes['_FillValue']
    elif 'missing_value' in self._attributes:
        missing_value = self._attributes['missing_value']
    else:
        missing_value = None
    return missing_value