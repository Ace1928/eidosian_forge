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
class RepartitionStage(AllToAllStage):
    """Implementation of `Dataset.repartition()`."""

    def __init__(self, num_blocks: int, shuffle: bool):
        if shuffle:

            def do_shuffle(block_list, ctx: TaskContext, clear_input_blocks: bool, block_udf, remote_args):
                if clear_input_blocks:
                    blocks = block_list.copy()
                    block_list.clear()
                else:
                    blocks = block_list
                context = DataContext.get_current()
                if context.use_push_based_shuffle:
                    shuffle_op_cls = PushBasedShufflePartitionOp
                else:
                    shuffle_op_cls = SimpleShufflePartitionOp
                shuffle_op = shuffle_op_cls(block_udf, random_shuffle=False)
                return shuffle_op.execute(blocks, num_blocks, clear_input_blocks, map_ray_remote_args=remote_args, reduce_ray_remote_args=remote_args, ctx=ctx)
            super().__init__('Repartition', num_blocks, do_shuffle, supports_block_udf=True, sub_stage_names=['ShuffleMap', 'ShuffleReduce'])
        else:

            def do_fast_repartition(block_list, ctx: TaskContext, clear_input_blocks: bool, *_):
                if clear_input_blocks:
                    blocks = block_list.copy()
                    block_list.clear()
                else:
                    blocks = block_list
                return fast_repartition(blocks, num_blocks, ctx)
            super().__init__('Repartition', num_blocks, do_fast_repartition, sub_stage_names=['Repartition'])