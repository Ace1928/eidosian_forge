import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
def __concat_key_with_prefixes(self, original_key):
    """Concat the necessary prefixes and key for isolation purpose for
        different jobs and different groups."""
    return f'{self._job_id.hex()}-{self._group_name}-{original_key}'