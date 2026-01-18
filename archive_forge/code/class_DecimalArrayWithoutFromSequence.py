from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        raise KeyError('For the test')