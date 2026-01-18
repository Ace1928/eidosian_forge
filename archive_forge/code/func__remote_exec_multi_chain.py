from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@ray.remote
def _remote_exec_multi_chain(num_returns: int, *args: Tuple, remote_executor=_REMOTE_EXEC) -> Generator:
    """
    Execute the deconstructed chain with a multiple return values in a worker process.

    Parameters
    ----------
    num_returns : int
        The number of return values.
    *args : tuple
        A deconstructed chain to be executed.
    remote_executor : _RemoteExecutor, default: _REMOTE_EXEC
        Do not change, it's used to avoid excessive serializations.

    Returns
    -------
    Generator
    """
    return remote_executor.construct(num_returns, args)