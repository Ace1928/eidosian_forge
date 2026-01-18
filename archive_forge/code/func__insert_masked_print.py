import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def _insert_masked_print(self):
    """
        Replace masked values with masked_print_option, casting all innermost
        dtypes to object.
        """
    if masked_print_option.enabled():
        mask = self._mask
        if mask is nomask:
            res = self._data
        else:
            data = self._data
            print_width = self._print_width if self.ndim > 1 else self._print_width_1d
            for axis in range(self.ndim):
                if data.shape[axis] > print_width:
                    ind = print_width // 2
                    arr = np.split(data, (ind, -ind), axis=axis)
                    data = np.concatenate((arr[0], arr[2]), axis=axis)
                    arr = np.split(mask, (ind, -ind), axis=axis)
                    mask = np.concatenate((arr[0], arr[2]), axis=axis)
            rdtype = _replace_dtype_fields(self.dtype, 'O')
            res = data.astype(rdtype)
            _recursive_printoption(res, mask, masked_print_option)
    else:
        res = self.filled(self.fill_value)
    return res