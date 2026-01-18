import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
def _skip_if_different_combine(self, data):
    if data.fill_value == 0:
        pytest.skip('Incorrected expected from Series.combine and tested elsewhere')