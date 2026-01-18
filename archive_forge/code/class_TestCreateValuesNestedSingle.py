import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestCreateValuesNestedSingle(CreateValues):
    """Check the creation of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr
    multiple_rows = 0
    _buffer = NbufferT[0]