import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
class TestNDArrayMethods:

    def test_repr(self):

        class MyArray(np.ndarray):

            def __array_function__(*args, **kwargs):
                return NotImplemented
        array = np.array(1).view(MyArray)
        assert_equal(repr(array), 'MyArray(1)')
        assert_equal(str(array), '1')