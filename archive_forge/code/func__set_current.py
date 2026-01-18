import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import ray
from ray._private.ray_constants import env_integer
from ray.util.annotations import DeveloperAPI
from ray.util.scheduling_strategies import SchedulingStrategyT
@staticmethod
def _set_current(context: 'DataContext') -> None:
    """Set the current context in a remote worker.

        This is used internally by Dataset to propagate the driver context to
        remote workers used for parallelization.
        """
    global _default_context
    _default_context = context