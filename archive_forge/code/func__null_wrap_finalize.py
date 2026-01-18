from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _null_wrap_finalize(finalize: Callable[[AggType], AggType]) -> Callable[[WrappedAggType], U]:
    """
    Wrap finalizer with null handling.

    If the accumulation is empty or None, the returned finalizer returns None.

    Args:
        finalize: The core finalizing function to wrap.

    Returns:
        A new finalizing function that handles nulls.
    """

    def _finalize(a: AggType) -> U:
        if a is None:
            return None
        a, has_data = _unwrap_acc(a)
        if not has_data:
            return None
        return finalize(a)
    return _finalize