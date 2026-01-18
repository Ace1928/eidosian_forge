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
class GcsAioLogSubscriber(_AioSubscriber):

    def __init__(self, worker_id: bytes=None, address: str=None, channel: grpc.Channel=None):
        super().__init__(pubsub_pb2.RAY_LOG_CHANNEL, worker_id, address, channel)

    async def poll(self, timeout=None) -> dict:
        """Polls for new log message.

        Returns:
            A dict containing a batch of log lines and their metadata,
            or None if polling times out or subscriber closed.
        """
        await self._poll(timeout=timeout)
        return self._pop_log_batch(self._queue)