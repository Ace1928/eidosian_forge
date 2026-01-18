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
def _is_share_resources_enabled(self, resource_name: str, enable_sharing_env: str):
    """Whether to share resource IDs on all workers
        based on enable_sharing_env.

        This will return true if resources are requested and greater than 0.
        Also, user can disable by configuring the `enable_sharing_env` to "0".

        Args:
            resource_name: The name of the resource/accelerator.
            enable_sharing_env: The name of the environment variable
                to check.
        """
    has_resource_requested = self._additional_resources_per_worker.get(resource_name, 0) > 0
    return has_resource_requested and ray_constants.env_bool(enable_sharing_env, True)