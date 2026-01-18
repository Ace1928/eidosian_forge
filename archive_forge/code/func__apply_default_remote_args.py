import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
@staticmethod
def _apply_default_remote_args(ray_remote_args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply defaults to the actor creation remote args."""
    ray_remote_args = ray_remote_args.copy()
    if 'scheduling_strategy' not in ray_remote_args:
        ctx = DataContext.get_current()
        ray_remote_args['scheduling_strategy'] = ctx.scheduling_strategy
    if 'max_restarts' not in ray_remote_args:
        ray_remote_args['max_restarts'] = -1
    if 'max_task_retries' not in ray_remote_args and ray_remote_args.get('max_restarts') != 0:
        ray_remote_args['max_task_retries'] = -1
    return ray_remote_args