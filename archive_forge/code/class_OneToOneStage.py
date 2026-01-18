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
class OneToOneStage(Stage):
    """A stage that transforms blocks independently (e.g., map or filter)."""

    def __init__(self, name: str, block_fn: BlockTransform, compute: Union[str, ComputeStrategy], ray_remote_args: dict, min_rows_per_block: Optional[int]=None, fn: Optional[UserDefinedFunction]=None, fn_args: Optional[Iterable[Any]]=None, fn_kwargs: Optional[Dict[str, Any]]=None, fn_constructor_args: Optional[Iterable[Any]]=None, fn_constructor_kwargs: Optional[Dict[str, Any]]=None):
        super().__init__(name, None)
        self.block_fn = block_fn
        self.compute = compute or TaskPoolStrategy()
        self.ray_remote_args = ray_remote_args or {}
        self.min_rows_per_block = min_rows_per_block
        self.fn = fn
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs
        self.fn_constructor_args = fn_constructor_args
        self.fn_constructor_kwargs = fn_constructor_kwargs

    def can_fuse(self, prev: Stage):
        if not isinstance(prev, OneToOneStage):
            return False
        if is_task_compute(self.compute) and prev.compute != self.compute:
            return False
        if isinstance(self.fn, CallableClass) and isinstance(prev.fn, CallableClass) and (prev.fn != self.fn or (prev.fn_constructor_args != self.fn_constructor_args or prev.fn_constructor_kwargs != self.fn_constructor_kwargs)):
            return False
        if not _are_remote_args_compatible(prev.ray_remote_args, self.ray_remote_args):
            return False
        return True

    def fuse(self, prev: Stage):
        if not self.can_fuse(prev):
            raise ValueError(f'Tried to fuse {prev} with {self}, but these are not fusable.')
        name = prev.name + '->' + self.name
        prev_fn = prev.fn
        if isinstance(self.fn, CallableClass) and isinstance(prev_fn, CallableClass):
            assert self.fn == prev_fn
            assert prev.fn_constructor_args == self.fn_constructor_args and prev.fn_constructor_kwargs == self.fn_constructor_kwargs
            use_outer_fn = True
            prev_fn = None
        else:
            use_outer_fn = False
        fn_args, unpack_args = _pack_args(self.fn_args, self.fn_kwargs, prev.fn_args, prev.fn_kwargs)
        block_fn1 = prev.block_fn
        block_fn2 = self.block_fn
        if prev.min_rows_per_block is not None and self.min_rows_per_block is not None:
            min_rows_per_block = max(prev.min_rows_per_block, self.min_rows_per_block)
        elif prev.min_rows_per_block is not None:
            min_rows_per_block = prev.min_rows_per_block
        else:
            min_rows_per_block = self.min_rows_per_block

        def block_fn(blocks: Iterable[Block], ctx: TaskContext, fn: UserDefinedFunction, *fn_args, **fn_kwargs) -> Iterable[Block]:
            assert not fn_kwargs, fn_kwargs
            self_fn_args, self_fn_kwargs, prev_fn_args, prev_fn_kwargs = unpack_args(fn_args)
            self_fn_args = self_fn_args if fn is None else (fn,) + self_fn_args
            if use_outer_fn:
                prev_fn_ = fn
            else:
                prev_fn_ = prev_fn
            prev_fn_args = prev_fn_args if prev_fn_ is None else (prev_fn_,) + prev_fn_args
            blocks = block_fn1(blocks, ctx, *prev_fn_args, **prev_fn_kwargs)
            return block_fn2(blocks, ctx, *self_fn_args, **self_fn_kwargs)
        return OneToOneStage(name, block_fn, self.compute, prev.ray_remote_args, min_rows_per_block=min_rows_per_block, fn=self.fn, fn_args=fn_args, fn_kwargs={}, fn_constructor_args=self.fn_constructor_args, fn_constructor_kwargs=self.fn_constructor_kwargs)

    def __call__(self, blocks: BlockList, clear_input_blocks: bool, run_by_consumer: bool) -> Tuple[BlockList, dict]:
        compute = get_compute(self.compute)
        assert self.fn_constructor_args is None and self.fn_constructor_kwargs is None or isinstance(compute, ActorPoolStrategy)
        if blocks._owned_by_consumer:
            assert run_by_consumer, 'Blocks owned by consumer can only be consumed by consumer'
        blocks = compute._apply(self.block_fn, self.ray_remote_args, blocks, clear_input_blocks, name=self.name, min_rows_per_block=self.min_rows_per_block, fn=self.fn, fn_args=self.fn_args, fn_kwargs=self.fn_kwargs, fn_constructor_args=self.fn_constructor_args, fn_constructor_kwargs=self.fn_constructor_kwargs)
        assert isinstance(blocks, BlockList), blocks
        blocks._owned_by_consumer = run_by_consumer
        return (blocks, {})