import math
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.block_list import BlockList
from ray.data._internal.memory_tracing import trace_allocation
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import DatasetStats, _get_or_create_stats_actor
from ray.data._internal.util import _split_list
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import ReadTask
from ray.types import ObjectRef
def _submit_task(self, task_idx: int) -> Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]:
    """Submit the task with index task_idx.

        NOTE: When dynamic block splitting is enabled, returns
        Tuple[ObjectRef[DynamicObjectRefGenerator], None] instead of
        Tuple[ObjectRef[Block], ObjectRef[BlockMetadata]], and the blocks metadata will
        be fetched as the last element in DynamicObjectRefGenerator.
        """
    if self._stats_actor is None:
        self._stats_actor = _get_or_create_stats_actor()
    stats_actor = self._stats_actor
    if not self._execution_started:
        ray.get(stats_actor.record_start.remote(self._stats_uuid))
        self._execution_started = True
    task = self._tasks[task_idx]
    return (cached_remote_fn(_execute_read_task_split).options(num_returns='dynamic', **self._remote_args).remote(i=task_idx, task=task, context=DataContext.get_current(), stats_uuid=self._stats_uuid, stats_actor=stats_actor), None)