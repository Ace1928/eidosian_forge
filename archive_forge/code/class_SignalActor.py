import asyncio
import time
from typing import List
import numpy
import ray
import ray.experimental.internal_kv as internal_kv
from ray._raylet import GcsClient
from ray.util.collective.types import ReduceOp, torch_available
from ray.util.queue import _QueueActor
@ray.remote(num_cpus=0)
class SignalActor:

    def __init__(self, world_size):
        self.ready_events = [asyncio.Event() for _ in range(world_size)]
        self.world_size = world_size

    def send(self, rank, clear=False):
        self.ready_events[rank].set()
        if clear:
            self.ready_events[rank].clear()

    async def wait(self, should_wait=True):
        if should_wait:
            for i in range(self.world_size):
                await self.ready_events[i].wait()