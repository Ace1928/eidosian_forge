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
def _change_subscription(self, subscription: 'Subscription'):
    log.info('Updating subscribed topics to: %s', subscription.topics)
    if self._subscription is not None:
        self._subscription._unsubscribe()
    self._subscription = subscription
    self._notify_subscription_waiters()