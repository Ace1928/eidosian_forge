import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestClipmodeConverter(StringConverterTestCase):
    """ Tests of PyArray_ClipmodeConverter """
    conv = mt.run_clipmode_converter

    def test_valid(self):
        self._check('clip', 'NPY_CLIP')
        self._check('wrap', 'NPY_WRAP')
        self._check('raise', 'NPY_RAISE')
        assert self.conv(np.CLIP) == 'NPY_CLIP'
        assert self.conv(np.WRAP) == 'NPY_WRAP'
        assert self.conv(np.RAISE) == 'NPY_RAISE'