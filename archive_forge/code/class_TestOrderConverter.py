import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestOrderConverter(StringConverterTestCase):
    """ Tests of PyArray_OrderConverter """
    conv = mt.run_order_converter
    warn = False

    def test_valid(self):
        self._check('c', 'NPY_CORDER')
        self._check('f', 'NPY_FORTRANORDER')
        self._check('a', 'NPY_ANYORDER')
        self._check('k', 'NPY_KEEPORDER')

    def test_flatten_invalid_order(self):
        with pytest.raises(ValueError):
            self.conv('Z')
        for order in [False, True, 0, 8]:
            with pytest.raises(TypeError):
                self.conv(order)