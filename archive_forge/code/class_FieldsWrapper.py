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
class FieldsWrapper:
    """Wrapper to extract named fields from a dataset with a struct dtype"""
    extract_field = None

    def __init__(self, dset, prior_dtype, names):
        self._dset = dset
        if isinstance(names, str):
            self.extract_field = names
            names = [names]
        self.read_dtype = readtime_dtype(prior_dtype, names)

    def __array__(self, dtype=None):
        data = self[:]
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def __getitem__(self, args):
        data = self._dset.__getitem__(args, new_dtype=self.read_dtype)
        if self.extract_field is not None:
            data = data[self.extract_field]
        return data

    def __len__(self):
        """ Get the length of the underlying dataset

        >>> length = len(dataset.fields(['x', 'y']))
        """
        return len(self._dset)