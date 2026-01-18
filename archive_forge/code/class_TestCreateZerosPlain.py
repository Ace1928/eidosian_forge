import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
class TestCreateZerosPlain(CreateZeros):
    """Check the creation of heterogeneous arrays zero-valued (plain)"""
    _descr = Pdescr