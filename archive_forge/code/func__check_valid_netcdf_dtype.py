import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def _check_valid_netcdf_dtype(self, dtype):
    dtype = np.dtype(dtype)
    if dtype == bool:
        description = 'boolean'
    elif dtype == complex:
        description = 'complex'
    elif h5py.check_dtype(enum=dtype) is not None:
        description = 'enum'
    elif h5py.check_dtype(ref=dtype) is not None:
        description = 'reference'
    elif h5py.check_dtype(vlen=dtype) not in {None, str, bytes}:
        description = 'non-string variable length'
    else:
        description = None
    if description is not None:
        _invalid_netcdf_feature(f'{description} dtypes', self.invalid_netcdf)