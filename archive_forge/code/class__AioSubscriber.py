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
class _AioSubscriber(_SubscriberBase):
    """Async io subscriber to GCS.

    Usage example common to Aio subscribers:
        subscriber = GcsAioXxxSubscriber(address="...")
        await subscriber.subscribe()
        while running:
            ...... = await subscriber.poll()
            ......
        await subscriber.close()
    """

    def __init__(self, pubsub_channel_type, worker_id: bytes=None, address: str=None, channel: aiogrpc.Channel=None):
        super().__init__(worker_id)
        if address:
            assert channel is None, 'address and channel cannot both be specified'
            channel = gcs_utils.create_gcs_channel(address, aio=True)
        else:
            assert channel is not None, 'One of address and channel must be specified'
        self._stub = gcs_service_pb2_grpc.InternalPubSubGcsServiceStub(channel)
        self._channel = pubsub_channel_type
        self._queue = deque()
        self._close = asyncio.Event()

    async def subscribe(self) -> None:
        """Registers a subscription for the subscriber's channel type.

        Before the registration, published messages in the channel will not be
        saved for the subscriber.
        """
        if self._close.is_set():
            return
        req = self._subscribe_request(self._channel)
        await self._stub.GcsSubscriberCommandBatch(req, timeout=30)

    async def _poll_call(self, req, timeout=None):
        return await self._stub.GcsSubscriberPoll(req, timeout=timeout)

    async def _poll(self, timeout=None) -> None:
        while len(self._queue) == 0:
            req = self._poll_request()
            poll = get_or_create_event_loop().create_task(self._poll_call(req, timeout=timeout))
            close = get_or_create_event_loop().create_task(self._close.wait())
            done, others = await asyncio.wait([poll, close], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            other_task = others.pop()
            if not other_task.done():
                other_task.cancel()
            if poll not in done or close in done:
                break
            try:
                self._last_batch_size = len(poll.result().pub_messages)
                if poll.result().publisher_id != self._publisher_id:
                    if self._publisher_id != '':
                        logger.debug(f'replied publisher_id {poll.result().publisher_id}different from {self._publisher_id}, this should only happens during gcs failover.')
                    self._publisher_id = poll.result().publisher_id
                    self._max_processed_sequence_id = 0
                for msg in poll.result().pub_messages:
                    if msg.sequence_id <= self._max_processed_sequence_id:
                        logger.warn(f'Ignoring out of order message {msg}')
                        continue
                    self._max_processed_sequence_id = msg.sequence_id
                    self._queue.append(msg)
            except grpc.RpcError as e:
                if self._should_terminate_polling(e):
                    return
                raise

    async def close(self) -> None:
        """Closes the subscriber and its active subscription."""
        if self._close.is_set():
            return
        self._close.set()
        req = self._unsubscribe_request(channels=[self._channel])
        try:
            await self._stub.GcsSubscriberCommandBatch(req, timeout=5)
        except Exception:
            pass
        self._stub = None