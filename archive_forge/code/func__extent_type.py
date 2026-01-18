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
@with_phil
def _extent_type(self):
    """Get extent type for this dataset - SIMPLE, SCALAR or NULL"""
    return self.id.get_space().get_simple_extent_type()