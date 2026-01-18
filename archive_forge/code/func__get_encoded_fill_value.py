import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _get_encoded_fill_value(self):
    """
        Returns the encoded fill value for this variable as bytes.

        This is taken from either the _FillValue attribute, or the default fill
        value for this variable's data type.
        """
    if '_FillValue' in self._attributes:
        fill_value = np.array(self._attributes['_FillValue'], dtype=self.data.dtype).tobytes()
        if len(fill_value) == self.itemsize():
            return fill_value
        else:
            return self._default_encoded_fill_value()
    else:
        return self._default_encoded_fill_value()