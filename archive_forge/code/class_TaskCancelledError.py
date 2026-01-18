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
class TaskCancelledError(RayError):
    """Raised when this task is cancelled.

    Args:
        task_id: The TaskID of the function that was directly
            cancelled.
    """

    def __init__(self, task_id: Optional[TaskID]=None, error_message: Optional[str]=None):
        self.task_id = task_id
        self.error_message = error_message

    def __str__(self):
        msg = ''
        if self.task_id:
            msg = 'Task: ' + str(self.task_id) + ' was cancelled. '
        if self.error_message:
            msg += self.error_message
        return msg