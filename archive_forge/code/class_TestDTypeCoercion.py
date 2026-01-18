import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestDTypeCoercion(_DeprecationTestCase):
    message = 'Converting .* to a dtype .*is deprecated'
    deprecated_types = [np.generic, np.flexible, np.number, np.inexact, np.floating, np.complexfloating, np.integer, np.unsignedinteger, np.signedinteger, np.character]

    def test_dtype_coercion(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.dtype, args=(scalar_type,))

    def test_array_construction(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.array, args=([], scalar_type))

    def test_not_deprecated(self):
        for group in np.sctypes.values():
            for scalar_type in group:
                self.assert_not_deprecated(np.dtype, args=(scalar_type,))
        for scalar_type in [type, dict, list, tuple]:
            self.assert_not_deprecated(np.dtype, args=(scalar_type,))