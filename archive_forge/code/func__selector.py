import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
@property
def _selector(self):
    """Internal object for optimised selection of data"""
    if '_selector' in self._cache_props:
        return self._cache_props['_selector']
    slr = _selector.Selector(self.id.get_space())
    if self._readonly:
        self._cache_props['_selector'] = slr
    return slr