import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestCreateValuesPlainMultiple(CreateValues):
    """Check the creation of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr
    multiple_rows = 1
    _buffer = PbufferT