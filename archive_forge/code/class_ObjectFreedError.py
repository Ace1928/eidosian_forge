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
@DeveloperAPI
class ObjectFreedError(ObjectLostError):
    """Indicates that an object was manually freed by the application.

    Attributes:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        return self._base_str() + '\n\n' + 'The object was manually freed using the internal `free` call. Please ensure that `free` is only called once the object is no longer needed.'