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
class AstypeWrapper:
    """Wrapper to convert data on reading from a dataset.
    """

    def __init__(self, dset, dtype):
        self._dset = dset
        self._dtype = numpy.dtype(dtype)

    def __getitem__(self, args):
        return self._dset.__getitem__(args, new_dtype=self._dtype)

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.astype('f8'))
        """
        return len(self._dset)

    def __array__(self, dtype=None):
        data = self[:]
        if dtype is not None:
            data = data.astype(dtype)
        return data