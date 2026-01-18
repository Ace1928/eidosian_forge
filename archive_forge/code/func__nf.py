from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
def _nf(x):
    return f(x, [3])