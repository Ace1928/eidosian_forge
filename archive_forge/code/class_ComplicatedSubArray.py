import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class ComplicatedSubArray(SubArray):

    def __str__(self):
        return f'myprefix {self.view(SubArray)} mypostfix'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self}>'

    def _validate_input(self, value):
        if not isinstance(value, ComplicatedSubArray):
            raise ValueError('Can only set to MySubArray values')
        return value

    def __setitem__(self, item, value):
        super().__setitem__(item, self._validate_input(value))

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if not isinstance(value, np.ndarray):
            value = value.__array__().view(ComplicatedSubArray)
        return value

    @property
    def flat(self):
        return CSAIterator(self)

    @flat.setter
    def flat(self, value):
        y = self.ravel()
        y[:] = value

    def __array_wrap__(self, obj, context=None):
        obj = super().__array_wrap__(obj, context)
        if context is not None and context[0] is np.multiply:
            obj.info['multiplied'] = obj.info.get('multiplied', 0) + 1
        return obj