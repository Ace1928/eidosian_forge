from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)