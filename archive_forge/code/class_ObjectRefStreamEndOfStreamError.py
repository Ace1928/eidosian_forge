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
class ObjectRefStreamEndOfStreamError(RayError):
    """Raised by streaming generator tasks when there are no more ObjectRefs to
    read.
    """
    pass