import os
import sys
import warnings
from typing import Optional
import psutil
import ray
from packaging import version
from modin.config import (
from modin.core.execution.utils import set_env
from modin.error_message import ErrorMessage
from .engine_wrapper import ObjectRefTypes, RayWrapper
def _get_object_store_memory() -> Optional[int]:
    """
    Get the object store memory we should start Ray with, in bytes.

    - If the ``Memory`` config variable is set, return that.
    - On Linux, take system memory from /dev/shm. On other systems use total
      virtual memory.
    - On Mac, never return more than Ray-specified upper limit.

    Returns
    -------
    Optional[int]
        The object store memory size in bytes, or None if we should use the Ray
        default.
    """
    object_store_memory = Memory.get()
    if object_store_memory is not None:
        return object_store_memory
    virtual_memory = psutil.virtual_memory().total
    if sys.platform.startswith('linux'):
        shm_fd = os.open('/dev/shm', os.O_RDONLY)
        try:
            shm_stats = os.fstatvfs(shm_fd)
            system_memory = shm_stats.f_bsize * shm_stats.f_bavail
            if system_memory / (virtual_memory / 2) < 0.99:
                warnings.warn(f'The size of /dev/shm is too small ({system_memory} bytes). The required size ' + f'at least half of RAM ({virtual_memory // 2} bytes). Please, delete files in /dev/shm or ' + 'increase size of /dev/shm with --shm-size in Docker. Also, you can can override the memory ' + 'size for each Ray worker (in bytes) to the MODIN_MEMORY environment variable.')
        finally:
            os.close(shm_fd)
    else:
        system_memory = virtual_memory
    bytes_per_gb = 1000000000.0
    object_store_memory = int(_OBJECT_STORE_TO_SYSTEM_MEMORY_RATIO * system_memory // bytes_per_gb * bytes_per_gb)
    if object_store_memory == 0:
        return None
    if sys.platform == 'darwin' and version.parse(ray.__version__) >= version.parse('1.3.0'):
        object_store_memory = min(object_store_memory, _MAC_OBJECT_STORE_LIMIT_BYTES)
    return object_store_memory