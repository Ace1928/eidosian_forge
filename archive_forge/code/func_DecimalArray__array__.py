from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def DecimalArray__array__(self, dtype=None):
    raise Exception('tried to convert a DecimalArray to a numpy array')