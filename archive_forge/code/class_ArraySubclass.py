from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
class ArraySubclass(np.ndarray):

    def __iter__(self):
        for value in super().__iter__():
            yield np.array(value)

    def __getitem__(self, item):
        return np.array(super().__getitem__(item))