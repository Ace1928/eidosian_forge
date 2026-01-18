from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar, Union
import numpy as np
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
class _SortOp(ShuffleOp):

    @staticmethod
    def map(idx: int, block: Block, output_num_blocks: int, boundaries: List[T], sort_key: SortKey) -> List[Union[BlockMetadata, Block]]:
        stats = BlockExecStats.builder()
        out = BlockAccessor.for_block(block).sort_and_partition(boundaries, sort_key)
        meta = BlockAccessor.for_block(block).get_metadata(input_files=None, exec_stats=stats.build())
        return out + [meta]

    @staticmethod
    def reduce(sort_key: SortKey, *mapper_outputs: List[Block], partial_reduce: bool=False) -> (Block, BlockMetadata):
        return BlockAccessor.for_block(mapper_outputs[0]).merge_sorted_blocks(mapper_outputs, sort_key)