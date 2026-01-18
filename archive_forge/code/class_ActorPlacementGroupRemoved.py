import os
from traceback import format_exception
from typing import Optional, Union
import colorama
import ray._private.ray_constants as ray_constants
import ray.cloudpickle as pickle
from ray._raylet import ActorID, TaskID, WorkerID
from ray.core.generated.common_pb2 import (
from ray.util.annotations import DeveloperAPI, PublicAPI
import setproctitle
@PublicAPI
class ActorPlacementGroupRemoved(RayError):
    """Raised when the corresponding placement group was removed."""

    def __str__(self):
        return 'The placement group corresponding to this Actor has been removed.'