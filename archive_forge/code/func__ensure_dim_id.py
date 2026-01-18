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
def _ensure_dim_id(self):
    """Set _Netcdf4Dimid"""
    if self.dimensions and (not self._h5ds.attrs.get('_Netcdf4Dimid', False)):
        dim = self._parent._all_h5groups[self.dimensions[0]]
        if '_Netcdf4Dimid' in dim.attrs:
            self._h5ds.attrs['_Netcdf4Dimid'] = dim.attrs['_Netcdf4Dimid']