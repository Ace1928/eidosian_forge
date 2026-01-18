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
@cached_property
def _fast_read_ok(self):
    """Is this dataset suitable for simple reading"""
    return self._extent_type == h5s.SIMPLE and isinstance(self.id.get_type(), (h5t.TypeIntegerID, h5t.TypeFloatID))