from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
@_nvml()
def _is_full_nvlink(rank, world_size):
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                if not link_state:
                    return False
            except pynvml.NVMLError as error:
                logger.info(f'NVLink detection failed with message "{str(error)}". This is normal if your machine has no NVLink equipped')
                return False
    return True