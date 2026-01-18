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
class ExternalLink:
    """
        Represents an HDF5 external link.  Paths may be absolute or relative.
        No checking is performed to ensure either the target or file exists.
    """

    @property
    def path(self):
        """ Soft link path, i.e. the part inside the HDF5 file. """
        return self._path

    @property
    def filename(self):
        """ Path to the external HDF5 file in the filesystem. """
        return self._filename

    def __init__(self, filename, path):
        self._filename = filename_decode(filename_encode(filename))
        self._path = path

    def __repr__(self):
        return '<ExternalLink to "%s" in file "%s"' % (self.path, self.filename)