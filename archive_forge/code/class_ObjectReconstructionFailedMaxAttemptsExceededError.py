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
class ObjectReconstructionFailedMaxAttemptsExceededError(ObjectLostError):
    """Indicates that the object cannot be reconstructed because the maximum
    number of task retries has been exceeded.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        return self._base_str() + '\n\n' + 'The object cannot be reconstructed because the maximum number of task retries has been exceeded. To prevent this error, set `@ray.remote(max_retries=<num retries>)` (default 3).'