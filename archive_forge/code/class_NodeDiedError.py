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
class NodeDiedError(RayError):
    """Indicates that the node is either dead or unreachable."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message