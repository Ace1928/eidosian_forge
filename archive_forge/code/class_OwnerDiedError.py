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
class OwnerDiedError(ObjectLostError):
    """Indicates that the owner of the object has died while there is still a
    reference to the object.

    Args:
        object_ref_hex: Hex ID of the object.
    """

    def __str__(self):
        log_loc = '`/tmp/ray/session_latest/logs`'
        if self.owner_address:
            try:
                addr = Address()
                addr.ParseFromString(self.owner_address)
                ip_addr = addr.ip_address
                worker_id = WorkerID(addr.worker_id)
                log_loc = f'`/tmp/ray/session_latest/logs/*{worker_id.hex()}*` at IP address {ip_addr}'
            except Exception:
                pass
        return self._base_str() + '\n\n' + f"The object's owner has exited. This is the Python worker that first created the ObjectRef via `.remote()` or `ray.put()`. Check cluster logs ({log_loc}) for more information about the Python worker failure."