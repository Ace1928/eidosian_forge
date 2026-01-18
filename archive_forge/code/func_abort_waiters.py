import logging
import contextlib
import copy
import time
from asyncio import shield, Event, Future
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Pattern, Set
from aiokafka.errors import IllegalStateError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.util import create_future, get_running_loop
def abort_waiters(self, exc):
    """ Critical error occurred, we will abort any pending waiter
        """
    for waiter in self._assignment_waiters:
        if not waiter.done():
            waiter.set_exception(copy.copy(exc))
    self._subscription_waiters.clear()
    for waiter in self._fetch_waiters:
        if not waiter.done():
            waiter.set_exception(copy.copy(exc))