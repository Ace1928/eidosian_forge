import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class CustomSeries(Series):

    @property
    def _constructor(self):
        return CustomSeries

    def custom_series_function(self):
        return 'OK'