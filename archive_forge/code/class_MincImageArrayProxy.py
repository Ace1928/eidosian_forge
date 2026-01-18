from __future__ import annotations
from numbers import Integral
import numpy as np
from .externals.netcdf import netcdf_file
from .fileslice import canonical_slicers
from .spatialimages import SpatialHeader, SpatialImage
class MincImageArrayProxy:
    """MINC implementation of array proxy protocol

    The array proxy allows us to freeze the passed fileobj and
    header such that it returns the expected data array.
    """

    def __init__(self, minc_file):
        self.minc_file = minc_file
        self._shape = minc_file.get_data_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_proxy(self):
        return True

    def __array__(self, dtype=None):
        """Read data from file and apply scaling, casting to ``dtype``

        If ``dtype`` is unspecified, the dtype is automatically determined.

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        arr = self.minc_file.get_scaled_data(sliceobj=())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, sliceobj):
        """Read slice `sliceobj` of data from file"""
        return self.minc_file.get_scaled_data(sliceobj)