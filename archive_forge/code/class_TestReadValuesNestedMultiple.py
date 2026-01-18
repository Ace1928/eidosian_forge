import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestReadValuesNestedMultiple(ReadValuesNested):
    """Check the values of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr
    multiple_rows = True
    _buffer = NbufferT