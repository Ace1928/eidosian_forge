import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def _warn_if_using_deprecated_placement_group(options: Dict[str, Any], caller_stacklevel: int):
    placement_group = options['placement_group']
    placement_group_bundle_index = options['placement_group_bundle_index']
    placement_group_capture_child_tasks = options['placement_group_capture_child_tasks']
    if placement_group != 'default':
        warnings.warn(f'placement_group parameter is deprecated. Use scheduling_strategy=PlacementGroupSchedulingStrategy(...) instead, see the usage at https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/package-ref.html#ray-remote.', DeprecationWarning, stacklevel=caller_stacklevel + 1)
    if placement_group_bundle_index != -1:
        warnings.warn(f'placement_group_bundle_index parameter is deprecated. Use scheduling_strategy=PlacementGroupSchedulingStrategy(...) instead, see the usage at https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/package-ref.html#ray-remote.', DeprecationWarning, stacklevel=caller_stacklevel + 1)
    if placement_group_capture_child_tasks:
        warnings.warn(f'placement_group_capture_child_tasks parameter is deprecated. Use scheduling_strategy=PlacementGroupSchedulingStrategy(...) instead, see the usage at https://docs.ray.io/en/{get_ray_doc_version()}/ray-core/package-ref.html#ray-remote.', DeprecationWarning, stacklevel=caller_stacklevel + 1)