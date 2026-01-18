import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
@staticmethod
def _sync_streams(device_list, events, streams):
    """Let NCCL streams wait for current streams for every device."""
    if ENV.NCCL_USE_MULTISTREAM.val:
        for i, device in enumerate(device_list):
            with nccl_util.Device(device):
                events[i].record(cupy.cuda.get_current_stream())
                streams[i].wait_event(events[i])