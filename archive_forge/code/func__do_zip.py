import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.fast_repartition import fast_repartition
from ray.data._internal.plan import AllToAllStage
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle_and_partition import (
from ray.data._internal.sort import SortKey, sort_impl
from ray.data._internal.split import _split_at_index, _split_at_indices
from ray.data.block import (
from ray.data.context import DataContext
def _do_zip(block: Block, *other_blocks: Block, inverted: bool=False) -> Tuple[Block, BlockMetadata]:
    stats = BlockExecStats.builder()
    builder = DelegatingBlockBuilder()
    for other_block in other_blocks:
        builder.add_block(other_block)
    other_block = builder.build()
    if inverted:
        block, other_block = (other_block, block)
    result = BlockAccessor.for_block(block).zip(other_block)
    br = BlockAccessor.for_block(result)
    return (result, br.get_metadata(input_files=[], exec_stats=stats.build()))