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
class CrossLanguageError(RayError):
    """Raised from another language."""

    def __init__(self, ray_exception):
        super().__init__('An exception raised from {}:\n{}'.format(Language.Name(ray_exception.language), ray_exception.formatted_exception_string))