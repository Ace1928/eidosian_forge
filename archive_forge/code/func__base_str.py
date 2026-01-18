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
def _base_str(self):
    msg = f'Failed to retrieve object {self.object_ref_hex}. '
    if self.call_site:
        msg += f'The ObjectRef was created at: {self.call_site}'
    else:
        msg += 'To see information about where this ObjectRef was created in Python, set the environment variable RAY_record_ref_creation_sites=1 during `ray start` and `ray.init()`.'
    return msg