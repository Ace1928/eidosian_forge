import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
class IntegerArrayNoCopy(pd.core.arrays.IntegerArray):

    def copy(self):
        assert False