import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.fixture
def _integer_integers(_integer):
    return _integer * np.random.default_rng(2).integers(0, 2, size=np.shape(_integer))