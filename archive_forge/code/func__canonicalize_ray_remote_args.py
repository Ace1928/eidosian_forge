import copy
import functools
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Set, Union
import ray
from ray import ObjectRef
from ray._raylet import ObjectRefGenerator
from ray.data._internal.compute import (
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.interfaces.physical_operator import (
from ray.data._internal.execution.operators.base_physical_operator import (
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _canonicalize_ray_remote_args(ray_remote_args: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce rules on ray remote args for map tasks.

    Namely, args must explicitly specify either CPU or GPU, not both. Disallowing
    mixed resources avoids potential starvation and deadlock issues during scheduling,
    and should not be a serious limitation for users.
    """
    ray_remote_args = ray_remote_args.copy()
    if 'num_cpus' not in ray_remote_args and 'num_gpus' not in ray_remote_args:
        ray_remote_args['num_cpus'] = 1
    if ray_remote_args.get('num_gpus', 0) > 0:
        if ray_remote_args.get('num_cpus', 0) != 0:
            raise ValueError('It is not allowed to specify both num_cpus and num_gpus for map tasks.')
    elif ray_remote_args.get('num_cpus', 0) > 0:
        if ray_remote_args.get('num_gpus', 0) != 0:
            raise ValueError('It is not allowed to specify both num_cpus and num_gpus for map tasks.')
    return ray_remote_args