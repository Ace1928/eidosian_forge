import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
class MMatrix(MaskedArray, np.matrix):

    def __new__(cls, data, mask=nomask):
        mat = np.matrix(data)
        _data = MaskedArray.__new__(cls, data=mat, mask=mask)
        return _data

    def __array_finalize__(self, obj):
        np.matrix.__array_finalize__(self, obj)
        MaskedArray.__array_finalize__(self, obj)
        return

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view