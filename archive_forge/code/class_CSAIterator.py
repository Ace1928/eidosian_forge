import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class CSAIterator:
    """
    Flat iterator object that uses its own setter/getter
    (works around ndarray.flat not propagating subclass setters/getters
    see https://github.com/numpy/numpy/issues/4564)
    roughly following MaskedIterator
    """

    def __init__(self, a):
        self._original = a
        self._dataiter = a.view(np.ndarray).flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        out = self._dataiter.__getitem__(indx)
        if not isinstance(out, np.ndarray):
            out = out.__array__()
        out = out.view(type(self._original))
        return out

    def __setitem__(self, index, value):
        self._dataiter[index] = self._original._validate_input(value)

    def __next__(self):
        return next(self._dataiter).__array__().view(type(self._original))