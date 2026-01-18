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
@staticmethod
def _compute_shuffle_schedule(num_cpus_per_node_map: Dict[str, int], num_input_blocks: int, merge_factor: int, num_output_blocks: int) -> _PushBasedShuffleStage:
    num_cpus_total = sum((v for v in num_cpus_per_node_map.values()))
    task_parallelism = min(num_cpus_total, num_input_blocks)
    num_tasks_per_map_merge_group = merge_factor + 1
    num_merge_tasks_per_round = 0
    merge_task_placement = []
    leftover_cpus = 0
    for node, num_cpus in num_cpus_per_node_map.items():
        node_parallelism = min(num_cpus, num_input_blocks // len(num_cpus_per_node_map))
        num_merge_tasks = node_parallelism // num_tasks_per_map_merge_group
        for i in range(num_merge_tasks):
            merge_task_placement.append(node)
        num_merge_tasks_per_round += num_merge_tasks
        leftover_cpus += node_parallelism % num_tasks_per_map_merge_group
        if num_merge_tasks == 0 and leftover_cpus > num_tasks_per_map_merge_group:
            merge_task_placement.append(node)
            num_merge_tasks_per_round += 1
            leftover_cpus -= num_tasks_per_map_merge_group
    if num_merge_tasks_per_round == 0:
        merge_task_placement.append(list(num_cpus_per_node_map)[0])
        num_merge_tasks_per_round = 1
    assert num_merge_tasks_per_round == len(merge_task_placement)
    num_map_tasks_per_round = max(task_parallelism - num_merge_tasks_per_round, 1)
    num_rounds = math.ceil(num_input_blocks / num_map_tasks_per_round)
    return _PushBasedShuffleStage(num_output_blocks, num_rounds, num_map_tasks_per_round, merge_task_placement)