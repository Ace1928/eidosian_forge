import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
class AllToAllStage(Stage):
    """A stage that transforms blocks holistically (e.g., shuffle)."""

    def __init__(self, name: str, num_blocks: Optional[int], fn: Callable[[BlockList, bool, Callable], Tuple[BlockList, dict]], supports_block_udf: bool=False, block_udf: Optional[BlockTransform]=None, remote_args: Optional[Dict[str, Any]]=None, sub_stage_names: Optional[List[str]]=None):
        super().__init__(name, num_blocks)
        self.fn = fn
        self.supports_block_udf = supports_block_udf
        self.block_udf = block_udf
        self.ray_remote_args = remote_args or {}
        self.sub_stage_names = sub_stage_names

    def can_fuse(self, prev: Stage):
        context = DataContext.get_current()
        if not context.optimize_fuse_shuffle_stages:
            return False
        if not self.supports_block_udf:
            return False
        if not isinstance(prev, OneToOneStage):
            return False
        if not is_task_compute(prev.compute):
            return False
        if not _are_remote_args_compatible(prev.ray_remote_args, self.ray_remote_args):
            return False
        return True

    def fuse(self, prev: Stage):
        if not self.can_fuse(prev):
            raise ValueError(f'Tried to fuse {prev} with {self}, but these are not fusable.')
        assert self.supports_block_udf
        assert prev.fn_constructor_args is None and prev.fn_constructor_kwargs is None
        name = prev.name + '->' + self.name
        prev_fn_args = prev.fn_args or tuple()
        prev_fn_args = prev_fn_args if prev.fn is None else (prev.fn,) + prev_fn_args
        prev_fn_kwargs = prev.fn_kwargs or {}
        prev_block_fn = prev.block_fn
        if self.block_udf is None:

            def block_udf(blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
                yield from prev_block_fn(blocks, ctx, *prev_fn_args, **prev_fn_kwargs)
        else:
            self_block_udf = self.block_udf

            def block_udf(blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
                blocks = prev_block_fn(blocks, ctx, *prev_fn_args, **prev_fn_kwargs)
                yield from self_block_udf(blocks, ctx)
        return AllToAllStage(name, self.num_blocks, self.fn, True, block_udf, prev.ray_remote_args, self.sub_stage_names)

    def __call__(self, blocks: BlockList, clear_input_blocks: bool, run_by_consumer: bool) -> Tuple[BlockList, dict]:
        from ray.data._internal.stage_impl import RandomizeBlocksStage
        in_blocks_owned_by_consumer = blocks._owned_by_consumer
        if in_blocks_owned_by_consumer:
            assert run_by_consumer, 'Blocks owned by consumer can only be consumed by consumer'
        blocks, stage_info = self.fn(blocks, clear_input_blocks, self.block_udf, self.ray_remote_args)
        assert isinstance(blocks, BlockList), blocks
        if isinstance(self, RandomizeBlocksStage):
            blocks._owned_by_consumer = in_blocks_owned_by_consumer
        else:
            blocks._owned_by_consumer = run_by_consumer
        return (blocks, stage_info)