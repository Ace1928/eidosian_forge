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
def _destroy_store(group_key):
    """Destroy the KV store (Ray named actor).

        Args:
            group_key: the unique key to retrieve the KV store.

        Returns:
            None
        """
    store_name = get_store_name(group_key)
    store = ray.get_actor(store_name)
    ray.kill(store)