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
def _get_or_compute(self, i: int) -> Tuple[ObjectRef[MaybeBlockPartition], Union[None, ObjectRef[BlockMetadata]]]:
    assert i < len(self._tasks), i
    if not self._block_partition_refs[i]:
        for j in range(max(i + 1, i * 2)):
            if j >= len(self._block_partition_refs):
                break
            if not self._block_partition_refs[j]:
                self._block_partition_refs[j], self._block_partition_meta_refs[j] = self._submit_task(j)
        assert self._block_partition_refs[i], self._block_partition_refs
    trace_allocation(self._block_partition_refs[i], f'LazyBlockList.get_or_compute({i})')
    return (self._block_partition_refs[i], self._block_partition_meta_refs[i])