import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def _test_not_equal(self, a, b):
    with assert_raises(AssertionError):
        self._assert_func(a, b)