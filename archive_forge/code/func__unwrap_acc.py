from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _unwrap_acc(a: WrappedAggType) -> Tuple[AggType, bool]:
    """
    Unwrap the accumulation, which we assume has been wrapped (via _wrap_acc) with a
    numeric boolean flag indicating whether or not this accumulation contains real data.

    Args:
        a: The wrapped accumulation value that we wish to unwrap.

    Returns:
        A tuple containing the unwrapped accumulation value and a boolean indicating
        whether the accumulation contains real data.
    """
    has_data = a[-1] == 1
    a = a[:-1]
    if len(a) == 1:
        a = a[0]
    return (a, has_data)