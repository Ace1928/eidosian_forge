import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
def _push_error_to_user(self, exc):
    """ Most critical errors are not something we can continue execution
        without user action. Well right now we just drop the Consumer, but
        java client would certainly be ok if we just poll another time, maybe
        it will need to rejoin, but not fail with GroupAuthorizationFailedError
        till the end of days...
        XXX: Research if we can't have the same error several times. For
             example if user gets GroupAuthorizationFailedError and adds
             permission for the group, would Consumer work right away or would
             still raise exception a few times?
        """
    exc = copy.copy(exc)
    self._subscription.abort_waiters(exc)
    self._pending_exception = exc
    self._error_consumed_fut = create_future()
    return asyncio.wait([self._error_consumed_fut, self._closing], return_when=asyncio.FIRST_COMPLETED)