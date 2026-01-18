import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import ray
from ray._private import ray_constants
from ray._private.utils import get_ray_doc_version
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (
def _check_deprecate_placement_group(options: Dict[str, Any]):
    """Check if deprecated placement group option exists."""
    placement_group = options.get('placement_group', 'default')
    scheduling_strategy = options.get('scheduling_strategy')
    if placement_group not in ('default', None) and scheduling_strategy is not None:
        raise ValueError('Placement groups should be specified via the scheduling_strategy option. The placement_group option is deprecated.')