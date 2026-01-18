import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
@property
def _scale_refs(self):
    """Return dimension scale references"""
    return list(self._h5ds.attrs.get('REFERENCE_LIST', []))