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
class OutOfDiskError(RayError):
    """Indicates that the local disk is full.

    This is raised if the attempt to store the object fails
    because both the object store and disk are full.
    """

    def __str__(self):
        return super(OutOfDiskError, self).__str__() + "\nThe object cannot be created because the local object store is full and the local disk's utilization is over capacity (95% by default).Tip: Use `df` on this node to check disk usage and `ray memory` to check object store memory usage."