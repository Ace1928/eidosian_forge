import itertools
from typing import List, Tuple
import ray
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split import _split_at_indices
from ray.data._internal.stats import StatsDict
from ray.data.block import (
def _get_num_rows_and_bytes(block: Block) -> Tuple[int, int]:
    block = BlockAccessor.for_block(block)
    return (block.num_rows(), block.size_bytes())