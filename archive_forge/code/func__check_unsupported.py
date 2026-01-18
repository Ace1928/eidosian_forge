import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
def _check_unsupported(self, data):
    if data.dtype == SparseDtype(int, 0):
        pytest.skip("Can't store nan in int array.")