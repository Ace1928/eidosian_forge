import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
def _parse_data(self, data):
    self._data = {}
    for name, value in data.items():
        if name.endswith('.'):
            if 'ndarray' in value:
                shape, dtype, offset = value['ndarray']
                dtype = dtype.encode()
                value = NDArrayReader(self._fd, shape, np.dtype(dtype), offset, self._little_endian)
            else:
                value = Reader(self._fd, data=value, _little_endian=self._little_endian)
            name = name[:-1]
        self._data[name] = value