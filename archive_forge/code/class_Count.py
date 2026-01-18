import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
@PublicAPI
class Count(AggregateFn):
    """Defines count aggregation."""

    def __init__(self):
        super().__init__(init=lambda k: 0, accumulate_block=lambda a, block: a + BlockAccessor.for_block(block).num_rows(), merge=lambda a1, a2: a1 + a2, name='count()')