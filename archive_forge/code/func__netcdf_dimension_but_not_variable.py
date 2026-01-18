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
def _netcdf_dimension_but_not_variable(h5py_dataset):
    return NOT_A_VARIABLE in h5py_dataset.attrs.get('NAME', b'')