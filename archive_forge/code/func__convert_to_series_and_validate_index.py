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
def _convert_to_series_and_validate_index(value_or_series: Union[bool, NumberT, pd.Series], index: pd.Index) -> pd.Series:
    """Returns a pd.Series of the given index with the corresponding values.

    Args:
      value_or_series: the values to be converted (if applicable).
      index: the index of the resulting pd.Series.

    Returns:
      pd.Series: The set of values with the given index.

    Raises:
      TypeError: If the type of `value_or_series` is not recognized.
      ValueError: If the index does not match.
    """
    if mbn.is_a_number(value_or_series) or isinstance(value_or_series, bool):
        result = pd.Series(data=value_or_series, index=index)
    elif isinstance(value_or_series, pd.Series):
        if value_or_series.index.equals(index):
            result = value_or_series
        else:
            raise ValueError('index does not match')
    else:
        raise TypeError('invalid type={}'.format(type(value_or_series)))
    return result