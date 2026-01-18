import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class MSubArray(SubArray, MaskedArray):

    def __new__(cls, data, info={}, mask=nomask):
        subarr = SubArray(data, info)
        _data = MaskedArray.__new__(cls, data=subarr, mask=mask)
        _data.info = subarr.info
        return _data

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view