from typing import List, Optional, Tuple, Union
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.sort import SortKey
from ray.data._internal.table_block import TableBlockAccessor
from ray.data.aggregate import AggregateFn, Count
from ray.data.aggregate._aggregate import _AggregateOnKeyBase
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata, KeyType
@staticmethod
def _prune_unused_columns(block: Block, key: Union[str, List[str]], aggs: Tuple[AggregateFn]) -> Block:
    """Prune unused columns from block before aggregate."""
    prune_columns = True
    columns = set()
    if isinstance(key, str):
        columns.add(key)
    elif isinstance(key, list):
        columns.update(key)
    elif callable(key):
        prune_columns = False
    for agg in aggs:
        if isinstance(agg, _AggregateOnKeyBase) and isinstance(agg._key_fn, str):
            columns.add(agg._key_fn)
        elif not isinstance(agg, Count):
            prune_columns = False
    block_accessor = BlockAccessor.for_block(block)
    if prune_columns and isinstance(block_accessor, TableBlockAccessor) and (block_accessor.num_rows() > 0):
        return block_accessor.select(list(columns))
    else:
        return block