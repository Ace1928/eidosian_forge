import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
@dataclass
class ResourceConfig:
    """
    Resource configuration for resource_ids to share between workers.

    Args:
        resource_name: The name of the resource to configure
         (Example: "neuron_cores" or "gpu").
        resource_enable_sharing_env_var: The environment variable to
         check if the resource should be shared.
        share_resource_ids_env_var: The environment variable to configure for
         sharing the resources with other workers.
    """
    resource_name: str
    resource_enable_sharing_env_var: str
    share_resource_ids_env_var: str