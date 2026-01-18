import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestCastingConverter(StringConverterTestCase):
    """ Tests of PyArray_CastingConverter """
    conv = mt.run_casting_converter
    case_insensitive = False
    exact_match = True

    def test_valid(self):
        self._check('no', 'NPY_NO_CASTING')
        self._check('equiv', 'NPY_EQUIV_CASTING')
        self._check('safe', 'NPY_SAFE_CASTING')
        self._check('same_kind', 'NPY_SAME_KIND_CASTING')
        self._check('unsafe', 'NPY_UNSAFE_CASTING')