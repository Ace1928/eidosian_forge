import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def _attribute_series(*, func: Callable[[IntVar], IntegralT], values: _IndexOrSeries) -> pd.Series:
    """Returns the attributes of `values`.

    Args:
      func: The function to call for getting the attribute data.
      values: The values that the function will be applied (element-wise) to.

    Returns:
      pd.Series: The attribute values.
    """
    return pd.Series(data=[func(v) for v in values], index=_get_index(values))