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
def _attach_dim_scales(self):
    """Attach dimension scales"""
    for n, dim in enumerate(self.dimensions):
        self._h5ds.dims[n].attach_scale(self._parent._all_dimensions[dim]._h5ds)