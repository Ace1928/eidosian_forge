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
class _PushBasedShuffleStage:

    def __init__(self, output_num_blocks: int, num_rounds: int, num_map_tasks_per_round: int, merge_task_placement: List[str]):
        self.num_rounds = num_rounds
        self.num_map_tasks_per_round = num_map_tasks_per_round
        node_strategies = {node_id: {'scheduling_strategy': NodeAffinitySchedulingStrategy(node_id, soft=True)} for node_id in set(merge_task_placement)}
        self._merge_task_options = [node_strategies[node_id] for node_id in merge_task_placement]
        self.merge_schedule = _MergeTaskSchedule(output_num_blocks, len(merge_task_placement))

    def get_merge_task_options(self, merge_idx):
        return self._merge_task_options[merge_idx]