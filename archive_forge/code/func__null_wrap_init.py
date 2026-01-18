from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _null_wrap_init(init: Callable[[KeyType], AggType]) -> Callable[[KeyType], WrappedAggType]:
    """
    Wraps an accumulation initializer with null handling.

    The returned initializer function adds on a has_data field that the accumulator
    uses to track whether an aggregation is empty.

    Args:
        init: The core init function to wrap.

    Returns:
        A new accumulation initializer function that can handle nulls.
    """

    def _init(k: KeyType) -> AggType:
        a = init(k)
        return _wrap_acc(a, has_data=False)
    return _init