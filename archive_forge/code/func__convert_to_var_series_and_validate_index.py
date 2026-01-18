import abc
import dataclasses
import math
import numbers
import typing
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from numpy import typing as npt
import pandas as pd
from ortools.linear_solver import linear_solver_pb2
from ortools.linear_solver.python import model_builder_helper as mbh
from ortools.linear_solver.python import model_builder_numbers as mbn
def _convert_to_var_series_and_validate_index(var_or_series: Union['Variable', pd.Series], index: pd.Index) -> pd.Series:
    """Returns a pd.Series of the given index with the corresponding values.

    Args:
      var_or_series: the variables to be converted (if applicable).
      index: the index of the resulting pd.Series.

    Returns:
      pd.Series: The set of values with the given index.

    Raises:
      TypeError: If the type of `value_or_series` is not recognized.
      ValueError: If the index does not match.
    """
    if isinstance(var_or_series, Variable):
        result = pd.Series(data=var_or_series, index=index)
    elif isinstance(var_or_series, pd.Series):
        if var_or_series.index.equals(index):
            result = var_or_series
        else:
            raise ValueError('index does not match')
    else:
        raise TypeError('invalid type={}'.format(type(var_or_series)))
    return result