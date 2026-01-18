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
class RuntimeEnvSetupError(RayError):
    """Raised when a runtime environment fails to be set up.

    Args:
        error_message: The error message that explains
            why runtime env setup has failed.
    """

    def __init__(self, error_message: str=None):
        self.error_message = error_message

    def __str__(self):
        msgs = ['Failed to set up runtime environment.']
        if self.error_message:
            msgs.append(self.error_message)
        return '\n'.join(msgs)