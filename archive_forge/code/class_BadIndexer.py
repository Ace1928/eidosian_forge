import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
class BadIndexer(BaseIndexer):

    def get_window_bounds(self):
        return None