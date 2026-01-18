from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _null_wrap_merge(ignore_nulls: bool, merge: Callable[[AggType, AggType], AggType]) -> Callable[[WrappedAggType, WrappedAggType], WrappedAggType]:
    """
    Wrap merge function with null handling.

    The returned merge function expects a1 and a2 to be either None or of the form:
    a = [acc_data_1, ..., acc_data_2, has_data].

    This merges two accumulations subject to the following null rules:
    1. If a1 is empty and a2 is empty, return empty accumulation.
    2. If a1 (a2) is empty and a2 (a1) is None, return None.
    3. If a1 (a2) is empty and a2 (a1) is non-None, return a2 (a1).
    4. If a1 (a2) is None, return a2 (a1) if ignoring nulls, None otherwise.
    5. If a1 and a2 are both non-null, return merge(a1, a2).

    Args:
        ignore_nulls: Whether nulls should be ignored or cause a None result.
        merge: The core merge function to wrap.

    Returns:
        A new merge function that handles nulls.
    """

    def _merge(a1: WrappedAggType, a2: WrappedAggType) -> WrappedAggType:
        if a1 is None:
            return a2 if ignore_nulls else None
        unwrapped_a1, a1_has_data = _unwrap_acc(a1)
        if not a1_has_data:
            return a2
        if a2 is None:
            return a1 if ignore_nulls else None
        unwrapped_a2, a2_has_data = _unwrap_acc(a2)
        if not a2_has_data:
            return a1
        a = merge(unwrapped_a1, unwrapped_a2)
        return _wrap_acc(a, has_data=True)
    return _merge