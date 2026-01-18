import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
def _attach_scale(self, refs):
    """Attach dimension scale to references"""
    for var, dim in refs:
        self._parent._all_h5groups[var].dims[dim].attach_scale(self._h5ds)