from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _null_wrap_accumulate_row(ignore_nulls: bool, on_fn: Callable[[T], T], accum: Callable[[AggType, T], AggType]) -> Callable[[WrappedAggType, T], WrappedAggType]:
    """
    Wrap accumulator function with null handling.

    The returned accumulate function expects a to be either None or of the form:
    a = [acc_data_1, ..., acc_data_n, has_data].

    This performs an accumulation subject to the following null rules:
    1. If r is null and ignore_nulls=False, return None.
    2. If r is null and ignore_nulls=True, return a.
    3. If r is non-null and a is None, return None.
    4. If r is non-null and a is non-None, return accum(a[:-1], r).

    Args:
        ignore_nulls: Whether nulls should be ignored or cause a None result.
        on_fn: Function selecting a subset of the row to apply the aggregation.
        accum: The core accumulator function to wrap.

    Returns:
        A new accumulator function that handles nulls.
    """

    def _accum(a: WrappedAggType, r: T) -> WrappedAggType:
        r = on_fn(r)
        if _is_null(r):
            if ignore_nulls:
                return a
            else:
                return None
        elif a is None:
            return None
        else:
            a, _ = _unwrap_acc(a)
            a = accum(a, r)
            return _wrap_acc(a, has_data=True)
    return _accum