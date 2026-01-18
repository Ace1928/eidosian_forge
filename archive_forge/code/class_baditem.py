from itertools import permutations, product
import pytest
from pytest import param
import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters
from numpy.testing import (
class baditem:

    def __len__(self):
        obj[0][0] = 2
        raise ValueError('not actually a sequence!')

    def __getitem__(self):
        pass