import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
def _to_on_fn(on: Optional[str]):
    if on is None:
        return lambda r: r
    elif isinstance(on, str):
        return lambda r: r[on]
    else:
        return on