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
def _transform_1d_boolean_indexers(key):
    """Find and transform 1D boolean indexers to int"""
    try:
        key = [np.asanyarray(k).nonzero()[0] if isinstance(k, (np.ndarray, list)) and type(k[0]) in (bool, np.bool_) else k for k in key]
    except TypeError:
        return key
    return tuple(key)