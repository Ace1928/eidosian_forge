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
def _lookup_dimensions(self):
    attrs = self._h5ds.attrs
    if '_Netcdf4Coordinates' in attrs and attrs.get('CLASS', None) == b'DIMENSION_SCALE':
        order_dim = {value._dimid: key for key, value in self._parent._all_dimensions.items()}
        return tuple((order_dim[coord_id] for coord_id in attrs['_Netcdf4Coordinates']))
    if 'DIMENSION_LIST' in attrs:
        if _unlabeled_dimension_mix(self._h5ds) == 'labeled':
            return tuple((self._root._h5file[ref[-1]].name.split('/')[-1] for ref in list(self._h5ds.attrs.get('DIMENSION_LIST', []))))
    child_name = self._h5ds.name.split('/')[-1]
    if child_name in self._parent._all_dimensions:
        return (child_name,)
    dims = []
    phony_dims = defaultdict(int)
    for axis, dim in enumerate(self._h5ds.dims):
        if len(dim):
            name = _name_from_dimension(dim)
        elif self._root._phony_dims_mode is None:
            raise ValueError(f"variable {self.name!r} has no dimension scale associated with axis {axis}. \nUse phony_dims='sort' for sorted naming or phony_dims='access' for per access naming.")
        else:
            dimsize = self._h5ds.shape[axis]
            dim_names = [d.name for d in self._parent._all_dimensions.maps[0].values() if d.size == dimsize]
            name = dim_names[phony_dims[dimsize]].split('/')[-1]
            phony_dims[dimsize] += 1
        dims.append(name)
    return tuple(dims)