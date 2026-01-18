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
class cls(RayTaskError, cause_cls):

    def __init__(self, cause):
        self.cause = cause
        self.args = (cause,)

    def __getattr__(self, name):
        return getattr(self.cause, name)

    def __str__(self):
        return error_msg