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
class AsStrWrapper:
    """Wrapper to decode strings on reading the dataset"""

    def __init__(self, dset, encoding, errors='strict'):
        self._dset = dset
        if encoding is None:
            encoding = h5t.check_string_dtype(dset.dtype).encoding
        self.encoding = encoding
        self.errors = errors

    def __getitem__(self, args):
        bytes_arr = self._dset[args]
        if numpy.isscalar(bytes_arr):
            return bytes_arr.decode(self.encoding, self.errors)
        return numpy.array([b.decode(self.encoding, self.errors) for b in bytes_arr.flat], dtype=object).reshape(bytes_arr.shape)

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.asstr())
        """
        return len(self._dset)

    def __array__(self):
        return numpy.array([b.decode(self.encoding, self.errors) for b in self._dset], dtype=object).reshape(self._dset.shape)