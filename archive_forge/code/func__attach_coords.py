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
def _attach_coords(self):
    dims = self.dimensions
    coord_ids = np.array([self._parent._all_dimensions[d]._dimid for d in dims], 'int32')
    if len(coord_ids) > 1:
        self._h5ds.attrs['_Netcdf4Coordinates'] = coord_ids