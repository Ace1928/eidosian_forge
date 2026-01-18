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
def _maybe_resize_dimensions(self, key, value):
    """Resize according to given (expanded) key with respect to variable dimensions"""
    new_shape = ()
    v = None
    for i, dim in enumerate(self.dimensions):
        if self._parent._all_dimensions[dim].isunlimited():
            if key[i].stop is None:
                if v is None:
                    v = np.asarray(value)
                if v.ndim == self.ndim:
                    new_max = max(v.shape[i], self._h5ds.shape[i])
                elif v.ndim == 0:
                    new_max = self._parent._all_dimensions[dim].size
                else:
                    raise IndexError('shape of data does not conform to slice')
            else:
                new_max = max(key[i].stop, self._h5ds.shape[i])
            if self._parent._all_dimensions[dim].size < new_max:
                self._parent.resize_dimension(dim, new_max)
            new_shape += (new_max,)
        else:
            new_shape += (self._parent._all_dimensions[dim].size,)
    if self._h5ds.shape != new_shape:
        self._h5ds.resize(new_shape)