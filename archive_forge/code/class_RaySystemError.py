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
class RaySystemError(RayError):
    """Indicates that Ray encountered a system error.

    This exception can be thrown when the raylet is killed.
    """

    def __init__(self, client_exc, traceback_str=None):
        self.client_exc = client_exc
        self.traceback_str = traceback_str

    def __str__(self):
        error_msg = f'System error: {self.client_exc}'
        if self.traceback_str:
            error_msg += f'\ntraceback: {self.traceback_str}'
        return error_msg