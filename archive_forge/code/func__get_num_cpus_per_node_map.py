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
def _get_num_cpus_per_node_map() -> Dict[str, int]:
    nodes = ray.nodes()
    num_cpus_per_node_map = {}
    for node in nodes:
        resources = node['Resources']
        num_cpus = int(resources.get('CPU', 0))
        if num_cpus == 0:
            continue
        num_cpus_per_node_map[node['NodeID']] = num_cpus
    return num_cpus_per_node_map