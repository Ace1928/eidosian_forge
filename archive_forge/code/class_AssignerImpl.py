import asyncio
from typing import Optional, Set
import logging
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
from google.cloud.pubsublite.internal.wire.assigner import Assigner
from google.cloud.pubsublite.internal.wire.retrying_connection import (
from google.api_core.exceptions import FailedPrecondition, GoogleAPICallError
from google.cloud.pubsublite.internal.wire.connection_reinitializer import (
from google.cloud.pubsublite.internal.wire.connection import Connection
from google.cloud.pubsublite.types.partition import Partition
from google.cloud.pubsublite_v1.types import (
class AssignerImpl(Assigner, ConnectionReinitializer[PartitionAssignmentRequest, PartitionAssignment]):
    _initial: InitialPartitionAssignmentRequest
    _connection: RetryingConnection[PartitionAssignmentRequest, PartitionAssignment]
    _outstanding_assignment: bool
    _receiver: Optional[asyncio.Future]
    _new_assignment: 'asyncio.Queue[Set[Partition]]'

    def __init__(self, initial: InitialPartitionAssignmentRequest, factory: ConnectionFactory[PartitionAssignmentRequest, PartitionAssignment]):
        self._initial = initial
        self._connection = RetryingConnection(factory, self)
        self._outstanding_assignment = False
        self._receiver = None
        self._new_assignment = asyncio.Queue(maxsize=1)

    async def __aenter__(self):
        await self._connection.__aenter__()
        return self

    def _start_receiver(self):
        assert self._receiver is None
        self._receiver = asyncio.ensure_future(self._receive_loop())

    async def _stop_receiver(self):
        if self._receiver:
            self._receiver.cancel()
            await wait_ignore_errors(self._receiver)
            self._receiver = None

    async def _receive_loop(self):
        while True:
            response = await self._connection.read()
            if self._outstanding_assignment or not self._new_assignment.empty():
                self._connection.fail(FailedPrecondition('Received a duplicate assignment on the stream while one was outstanding.'))
                return
            self._outstanding_assignment = True
            partitions = set()
            for partition in response.partitions:
                partitions.add(Partition(partition))
            _LOGGER.info(f'Received new assignment with partitions: {partitions}.')
            self._new_assignment.put_nowait(partitions)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stop_receiver()
        await self._connection.__aexit__(exc_type, exc_val, exc_tb)

    async def stop_processing(self, error: GoogleAPICallError):
        await self._stop_receiver()
        self._outstanding_assignment = False
        while not self._new_assignment.empty():
            self._new_assignment.get_nowait()

    async def reinitialize(self, connection: Connection[PartitionAssignmentRequest, PartitionAssignment]):
        await connection.write(PartitionAssignmentRequest(initial=self._initial))
        self._start_receiver()

    async def get_assignment(self) -> Set[Partition]:
        if self._outstanding_assignment:
            try:
                await self._connection.write(PartitionAssignmentRequest(ack=PartitionAssignmentAck()))
                self._outstanding_assignment = False
            except GoogleAPICallError as e:
                _LOGGER.debug(f'Assignment ack attempt failed due to stream failure: {e}')
        return await self._connection.await_unless_failed(self._new_assignment.get())