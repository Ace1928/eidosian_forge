import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def _interpolation_search(arr: List[int], x: int) -> int:
    """
    Return the position i of a sorted array so that arr[i] <= x < arr[i+1]

    Args:
        arr (`List[int]`): non-empty sorted list of integers
        x (`int`): query

    Returns:
        `int`: the position i so that arr[i] <= x < arr[i+1]

    Raises:
        `IndexError`: if the array is empty or if the query is outside the array values
    """
    i, j = (0, len(arr) - 1)
    while i < j and arr[i] <= x < arr[j]:
        k = i + (j - i) * (x - arr[i]) // (arr[j] - arr[i])
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = (k + 1, j)
        else:
            i, j = (i, k)
    raise IndexError(f"Invalid query '{x}' for size {(arr[-1] if len(arr) else 'none')}.")