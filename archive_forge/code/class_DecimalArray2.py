import datetime
import decimal
import re
import numpy as np
import pytest
import pytz
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import register_extension_dtype
from pandas.arrays import (
from pandas.core.arrays import (
from pandas.tests.extension.decimal import (
class DecimalArray2(DecimalArray):

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(scalars, (pd.Series, pd.Index)):
            raise TypeError('scalars should not be of type pd.Series or pd.Index')
        return super()._from_sequence(scalars, dtype=dtype, copy=copy)