import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
def _iter_connected_components(self):
    """Return subpaths split at MOVETOs."""
    if self.codes is None:
        yield self
    else:
        idxs = np.append((self.codes == Path.MOVETO).nonzero()[0], len(self.codes))
        for sl in map(slice, idxs, idxs[1:]):
            yield Path._fast_from_codes_and_verts(self.vertices[sl], self.codes[sl], self)