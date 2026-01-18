import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.fixture
def _array_mixed2(_mixed2):
    return _mixed2['D'].values.copy()