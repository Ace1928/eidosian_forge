from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@ray.remote(num_returns=2)
def _remote_exec_single_chain(*args: Tuple, remote_executor=_REMOTE_EXEC) -> Generator:
    """
    Execute the deconstructed chain with a single return value in a worker process.

    Parameters
    ----------
    *args : tuple
        A deconstructed chain to be executed.
    remote_executor : _RemoteExecutor, default: _REMOTE_EXEC
        Do not change, it's used to avoid excessive serializations.

    Returns
    -------
    Generator
    """
    return remote_executor.construct(num_returns=2, args=args)