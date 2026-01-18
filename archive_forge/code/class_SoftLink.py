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
class SoftLink:
    """
        Represents a symbolic ("soft") link in an HDF5 file.  The path
        may be absolute or relative.  No checking is performed to ensure
        that the target actually exists.
    """

    @property
    def path(self):
        """ Soft link value.  Not guaranteed to be a valid path. """
        return self._path

    def __init__(self, path):
        self._path = str(path)

    def __repr__(self):
        return '<SoftLink to "%s">' % self.path