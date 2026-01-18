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
def _get_maximum_dimension_id(self):
    dimids = []

    def _dimids(name, obj):
        if obj.attrs.get('CLASS', None) == b'DIMENSION_SCALE':
            dimids.append(obj.attrs.get('_Netcdf4Dimid', -1))
    self._h5file.visititems(_dimids)
    return max(dimids) if dimids else -1