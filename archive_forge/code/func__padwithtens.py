import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def _padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 10
    vector[-pad_width[1]:] = 10