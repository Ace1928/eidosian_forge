import logging
import threading
import cupy
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.const import ENV
def _init_stream_pool():
    global _device_stream_pool_map
    for i in range(MAX_GPU_PER_ACTOR):
        _device_stream_pool_map[i] = StreamPool(i)