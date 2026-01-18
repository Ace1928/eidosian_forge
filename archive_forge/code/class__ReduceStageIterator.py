import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class _ReduceStageIterator:

    def __init__(self, stage: _PushBasedShuffleStage, shuffle_reduce, all_merge_results: List[List[List[ObjectRef]]], ray_remote_args, reduce_args: List[Any]):
        self._shuffle_reduce = shuffle_reduce
        self._stage = stage
        self._reduce_arg_blocks: List[Tuple[int, List[ObjectRef]]] = []
        self._ray_remote_args = ray_remote_args
        self._reduce_args = reduce_args
        for reduce_idx in self._stage.merge_schedule.round_robin_reduce_idx_iterator():
            merge_idx = self._stage.merge_schedule.get_merge_idx_for_reducer_idx(reduce_idx)
            reduce_arg_blocks = [merge_results.pop(0) for merge_results in all_merge_results[merge_idx]]
            self._reduce_arg_blocks.append((reduce_idx, reduce_arg_blocks))
        assert len(self._reduce_arg_blocks) == stage.merge_schedule.output_num_blocks
        for merge_idx, merge_results in enumerate(all_merge_results):
            assert all((len(merge_result) == 0 for merge_result in merge_results)), f'Reduce stage did not process outputs from merge tasks at index: {merge_idx}'
        self._reduce_results: List[Tuple[int, ObjectRef]] = []

    def __iter__(self):
        return self

    def __next__(self):
        if not self._reduce_arg_blocks:
            raise StopIteration
        reduce_idx, reduce_arg_blocks = self._reduce_arg_blocks.pop(0)
        merge_idx = self._stage.merge_schedule.get_merge_idx_for_reducer_idx(reduce_idx)
        block, meta = self._shuffle_reduce.options(**self._ray_remote_args, **self._stage.get_merge_task_options(merge_idx), num_returns=2).remote(*self._reduce_args, *reduce_arg_blocks, partial_reduce=False)
        self._reduce_results.append((reduce_idx, block))
        return meta

    def pop_reduce_results(self):
        reduce_results = self._reduce_results
        self._reduce_results = []
        return reduce_results