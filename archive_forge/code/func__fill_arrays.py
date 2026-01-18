from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def _fill_arrays(self):
    """
        Get all arguments in array form
        """
    for i, arg in enumerate(self.args):
        if self.is_device_array(arg):
            self.arrays[i] = self.as_device_array(arg)
        elif isinstance(arg, (int, float, complex, np.number)):
            self.scalarpos.append(i)
        else:
            self.arrays[i] = np.asarray(arg)