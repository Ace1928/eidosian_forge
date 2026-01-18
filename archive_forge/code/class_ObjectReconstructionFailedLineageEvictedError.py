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
class ObjectReconstructionFailedLineageEvictedError(ObjectLostError):
    """Indicates that the object cannot be reconstructed because its lineage
    was evicted due to memory pressure.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        return self._base_str() + '\n\n' + 'The object cannot be reconstructed because its lineage has been evicted to reduce memory pressure. To prevent this error, set the environment variable RAY_max_lineage_bytes=<bytes> (default 1GB) during `ray start`.'