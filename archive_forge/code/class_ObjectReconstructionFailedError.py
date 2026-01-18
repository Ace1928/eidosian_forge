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
class ObjectReconstructionFailedError(ObjectLostError):
    """Indicates that the object cannot be reconstructed.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        return self._base_str() + '\n\n' + 'The object cannot be reconstructed because it was created by an actor, ray.put() call, or its ObjectRef was created by a different worker.'