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
class ObjectStoreFullError(RayError):
    """Indicates that the object store is full.

    This is raised if the attempt to store the object fails
    because the object store is full even after multiple retries.
    """

    def __str__(self):
        return super(ObjectStoreFullError, self).__str__() + '\nThe local object store is full of objects that are still in scope and cannot be evicted. Tip: Use the `ray memory` command to list active objects in the cluster.'