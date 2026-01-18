import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def _generate_non_native_data(self, n, m):
    data = randn(n, m)
    data = self._neg_byteorder(data)
    assert_(not data.dtype.isnative)
    return data