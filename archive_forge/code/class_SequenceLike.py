import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class SequenceLike:

    def __index__(self):
        return 0

    def __len__(self):
        return 1

    def __getitem__(self, item):
        raise IndexError('Not possible')