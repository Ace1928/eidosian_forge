import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
class DummyDtype(ExtensionDtype):
    type = int

    def __init__(self, numeric) -> None:
        self._numeric = numeric

    @property
    def name(self):
        return 'Dummy'

    @property
    def _is_numeric(self):
        return self._numeric