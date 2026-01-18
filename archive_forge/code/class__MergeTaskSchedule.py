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
class _MergeTaskSchedule:

    def __init__(self, output_num_blocks: int, num_merge_tasks_per_round: int):
        self.output_num_blocks = output_num_blocks
        self.num_merge_tasks_per_round = num_merge_tasks_per_round
        self.merge_partition_size = output_num_blocks // num_merge_tasks_per_round
        self._partitions_with_extra_task = output_num_blocks % num_merge_tasks_per_round
        if self.merge_partition_size == 0:
            self.num_merge_tasks_per_round = self._partitions_with_extra_task
            self.merge_partition_size = 1
            self._partitions_with_extra_task = 0

    def get_num_reducers_per_merge_idx(self, merge_idx: int) -> int:
        """
        Each intermediate merge task will produce outputs for a partition of P
        final reduce tasks. This helper function returns P based on the merge
        task index.
        """
        assert merge_idx < self.num_merge_tasks_per_round
        partition_size = self.merge_partition_size
        if merge_idx < self._partitions_with_extra_task:
            partition_size += 1
        return partition_size

    def get_merge_idx_for_reducer_idx(self, reducer_idx: int) -> int:
        if reducer_idx < (self.merge_partition_size + 1) * self._partitions_with_extra_task:
            merge_idx = reducer_idx // (self.merge_partition_size + 1)
        else:
            reducer_idx -= (self.merge_partition_size + 1) * self._partitions_with_extra_task
            merge_idx = self._partitions_with_extra_task + reducer_idx // self.merge_partition_size
        assert merge_idx < self.num_merge_tasks_per_round
        return merge_idx

    def round_robin_reduce_idx_iterator(self):
        """
        When there are multiple nodes, merge tasks are spread throughout the
        cluster to improve load-balancing. Each merge task produces outputs for
        a contiguous partition of reduce tasks. This method creates an iterator
        that returns reduce task indices round-robin across the merge tasks.
        This can be used to submit reduce tasks in a way that spreads the load
        evenly across the cluster.
        """
        idx = 0
        round_idx = 0
        while idx < self.output_num_blocks:
            for merge_idx in range(self.num_merge_tasks_per_round):
                if merge_idx < self._partitions_with_extra_task:
                    reduce_idx = merge_idx * (self.merge_partition_size + 1)
                    partition_size = self.merge_partition_size + 1
                else:
                    reduce_idx = self._partitions_with_extra_task * (self.merge_partition_size + 1)
                    merge_idx -= self._partitions_with_extra_task
                    reduce_idx += merge_idx * self.merge_partition_size
                    partition_size = self.merge_partition_size
                if round_idx >= partition_size:
                    continue
                reduce_idx += round_idx
                yield reduce_idx
                idx += 1
            round_idx += 1