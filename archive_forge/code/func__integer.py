import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.fixture
def _integer():
    return DataFrame(np.random.default_rng(2).integers(1, 100, size=(10001, 4)), columns=list('ABCD'), dtype='int64')