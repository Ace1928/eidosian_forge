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
def _neg_byteorder(self, a):
    a = np.asarray(a)
    if sys.byteorder == 'little':
        a = a.astype(a.dtype.newbyteorder('>'))
    else:
        a = a.astype(a.dtype.newbyteorder('<'))
    return a