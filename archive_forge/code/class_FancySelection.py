import numpy as np
from .base import product
from .. import h5s, h5r, _selector
class FancySelection(Selection):
    """
        Implements advanced NumPy-style selection operations in addition to
        the standard slice-and-int behavior.

        Indexing arguments may be ints, slices, lists of indices, or
        per-axis (1D) boolean arrays.

        Broadcasting is not supported for these selections.
    """

    @property
    def mshape(self):
        return self._mshape

    @property
    def array_shape(self):
        return self._array_shape

    def __init__(self, shape, spaceid=None, mshape=None, array_shape=None):
        super().__init__(shape, spaceid)
        if mshape is None:
            mshape = self.shape
        if array_shape is None:
            array_shape = mshape
        self._mshape = mshape
        self._array_shape = array_shape

    def expand_shape(self, source_shape):
        if not source_shape == self.array_shape:
            raise TypeError('Broadcasting is not supported for complex selections')
        return source_shape

    def broadcast(self, source_shape):
        if not source_shape == self.array_shape:
            raise TypeError('Broadcasting is not supported for complex selections')
        yield self._id