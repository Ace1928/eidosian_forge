import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestSelectkindConverter(StringConverterTestCase):
    """ Tests of PyArray_SelectkindConverter """
    conv = mt.run_selectkind_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check('introselect', 'NPY_INTROSELECT')