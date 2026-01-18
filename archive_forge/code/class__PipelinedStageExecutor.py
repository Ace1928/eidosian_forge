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
class _PipelinedStageExecutor:

    def __init__(self, stage_iter, num_tasks_per_round: int, max_concurrent_rounds: int=1, progress_bar: Optional[ProgressBar]=None):
        self._stage_iter = stage_iter
        self._num_tasks_per_round = num_tasks_per_round
        self._max_concurrent_rounds = max_concurrent_rounds
        self._progress_bar = progress_bar
        self._rounds: List[List[ObjectRef]] = []
        self._task_idx = 0
        self._submit_round()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Submit one round of tasks. If we already have the max concurrent rounds
        in flight, first wait for the oldest round of tasks to finish.
        """
        prev_metadata = []
        if all((len(r) == 0 for r in self._rounds)):
            raise StopIteration
        if len(self._rounds) >= self._max_concurrent_rounds:
            prev_metadata_refs = self._rounds.pop(0)
            if prev_metadata_refs:
                if self._progress_bar is not None:
                    prev_metadata = self._progress_bar.fetch_until_complete(prev_metadata_refs)
                else:
                    prev_metadata = ray.get(prev_metadata_refs)
        self._submit_round()
        return prev_metadata

    def _submit_round(self):
        assert len(self._rounds) < self._max_concurrent_rounds
        task_round = []
        for _ in range(self._num_tasks_per_round):
            try:
                task_round.append(next(self._stage_iter))
            except StopIteration:
                break
        self._rounds.append(task_round)