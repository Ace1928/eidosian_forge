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
def _notify_subscription_waiters(self):
    for waiter in self._subscription_waiters:
        if not waiter.done():
            waiter.set_result(None)
    self._subscription_waiters.clear()