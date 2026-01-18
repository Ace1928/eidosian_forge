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
@property
def _track_order(self):
    if self._root._h5py.__name__ == 'h5pyd':
        return False
    from h5py.h5p import CRT_ORDER_INDEXED, CRT_ORDER_TRACKED
    gcpl = self._h5group.id.get_create_plist()
    attr_creation_order = gcpl.get_attr_creation_order()
    order_tracked = bool(attr_creation_order & CRT_ORDER_TRACKED)
    order_indexed = bool(attr_creation_order & CRT_ORDER_INDEXED)
    return order_tracked and order_indexed