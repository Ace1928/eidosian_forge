from __future__ import annotations
from functools import (
from typing import (
import warnings
import numpy as np
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.generic import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many
def _any_pandas_objects(terms) -> bool:
    """
    Check a sequence of terms for instances of PandasObject.
    """
    return any((isinstance(term.value, PandasObject) for term in terms))