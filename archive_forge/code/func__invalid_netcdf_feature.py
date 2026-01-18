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
def _invalid_netcdf_feature(feature, allow):
    if not allow:
        msg = f'{feature} are not a supported NetCDF feature, and are not allowed by h5netcdf unless invalid_netcdf=True.'
        raise CompatibilityError(msg)