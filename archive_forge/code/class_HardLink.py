from contextlib import contextmanager
import posixpath as pp
import numpy
from .compat import filename_decode, filename_encode
from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support
class HardLink:
    """
        Represents a hard link in an HDF5 file.  Provided only so that
        Group.get works in a sensible way.  Has no other function.
    """
    pass