import asyncio
from collections import deque
import logging
import random
from typing import Tuple, List
import grpc
from ray._private.utils import get_or_create_event_loop
import ray._private.gcs_utils as gcs_utils
import ray._private.logging_utils as logging_utils
from ray.core.generated.gcs_pb2 import ErrorTableData
from ray.core.generated import dependency_pb2
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated import gcs_service_pb2
from ray.core.generated import common_pb2
from ray.core.generated import pubsub_pb2
class GcsAioActorSubscriber(_AioSubscriber):

    def __init__(self, worker_id: bytes=None, address: str=None, channel: grpc.Channel=None):
        super().__init__(pubsub_pb2.GCS_ACTOR_CHANNEL, worker_id, address, channel)

    @property
    def queue_size(self):
        return len(self._queue)

    async def poll(self, timeout=None, batch_size=500) -> List[Tuple[bytes, str]]:
        """Polls for new actor message.

        Returns:
            A tuple of binary actor ID and actor table data.
        """
        await self._poll(timeout=timeout)
        return self._pop_actors(self._queue, batch_size=batch_size)