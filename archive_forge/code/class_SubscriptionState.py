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
class SubscriptionState:
    """ Intermediate bridge to coordinate work between Consumer, Coordinator
    and Fetcher primitives.

        The class is different from kafka-python's implementation to provide
    a more friendly way to interact in asynchronous paradigm. The changes
    focus on making the internals less mutable (subscription, topic state etc.)
    paired with futures for when those change.
        Before there was a lot of trouble if user say did a subscribe between
    yield statements of a rebalance or other critical IO operation.
    """
    _subscription_type = SubscriptionType.NONE
    _subscribed_pattern = None
    _subscription = None
    _listener = None

    def __init__(self, loop=None):
        if loop is None:
            loop = get_running_loop()
        self._loop = loop
        self._subscription_waiters = []
        self._assignment_waiters = []
        self._fetch_count = 0
        self._last_fetch_ended = time.monotonic()

    @property
    def subscription(self) -> 'Subscription':
        return self._subscription

    @property
    def subscribed_pattern(self) -> Pattern:
        return self._subscribed_pattern

    @property
    def listener(self) -> ConsumerRebalanceListener:
        return self._listener

    @property
    def topics(self):
        if self._subscription is not None:
            return self._subscription.topics
        return set()

    def assigned_partitions(self) -> Set[TopicPartition]:
        if self._subscription is None:
            return set()
        if self._subscription.assignment is None:
            return set()
        return self._subscription.assignment.tps

    @property
    def reassignment_in_progress(self):
        if self._subscription is None:
            return True
        return self._subscription._reassignment_in_progress

    def partitions_auto_assigned(self) -> bool:
        return self._subscription_type == SubscriptionType.AUTO_TOPICS or self._subscription_type == SubscriptionType.AUTO_PATTERN

    def is_assigned(self, tp: TopicPartition) -> bool:
        if self._subscription is None:
            return False
        if self._subscription.assignment is None:
            return False
        return tp in self._subscription.assignment.tps

    def _set_subscription_type(self, subscription_type: SubscriptionType):
        if self._subscription_type == SubscriptionType.NONE or self._subscription_type == subscription_type:
            self._subscription_type = subscription_type
        else:
            raise IllegalStateError('Subscription to topics, partitions and pattern are mutually exclusive')

    def _change_subscription(self, subscription: 'Subscription'):
        log.info('Updating subscribed topics to: %s', subscription.topics)
        if self._subscription is not None:
            self._subscription._unsubscribe()
        self._subscription = subscription
        self._notify_subscription_waiters()

    def _assigned_state(self, tp: TopicPartition) -> 'TopicPartitionState':
        assert self._subscription is not None
        assert self._subscription.assignment is not None
        tp_state = self._subscription.assignment.state_value(tp)
        if tp_state is None:
            raise IllegalStateError(f'No current assignment for partition {tp}')
        return tp_state

    def _notify_subscription_waiters(self):
        for waiter in self._subscription_waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._subscription_waiters.clear()

    def _notify_assignment_waiters(self):
        for waiter in self._assignment_waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._assignment_waiters.clear()

    def subscribe(self, topics: Set[str], listener=None):
        """ Subscribe to a list (or tuple) of topics

        Caller: Consumer.
        Affects: SubscriptionState.subscription
        """
        assert isinstance(topics, set)
        assert listener is None or isinstance(listener, ConsumerRebalanceListener)
        self._set_subscription_type(SubscriptionType.AUTO_TOPICS)
        self._change_subscription(Subscription(topics, loop=self._loop))
        self._listener = listener
        self._notify_subscription_waiters()

    def subscribe_pattern(self, pattern: Pattern, listener=None):
        """ Subscribe to all topics matching a regex pattern.
        Subsequent calls `subscribe_from_pattern()` by Coordinator will provide
        the actual subscription topics.

        Caller: Consumer.
        Affects: SubscriptionState.subscribed_pattern
        """
        assert hasattr(pattern, 'match'), 'Expected Pattern type'
        assert listener is None or isinstance(listener, ConsumerRebalanceListener)
        self._set_subscription_type(SubscriptionType.AUTO_PATTERN)
        self._subscribed_pattern = pattern
        self._listener = listener

    def assign_from_user(self, partitions: Iterable[TopicPartition]):
        """ Manually assign partitions. After this call automatic assignment
        will be impossible and will raise an ``IllegalStateError``.

        Caller: Consumer.
        Affects: SubscriptionState.subscription
        """
        self._set_subscription_type(SubscriptionType.USER_ASSIGNED)
        self._change_subscription(ManualSubscription(partitions, loop=self._loop))
        self._notify_assignment_waiters()

    def unsubscribe(self):
        """ Unsubscribe from the last subscription. This will also clear the
        subscription type.

        Caller: Consumer.
        Affects: SubscriptionState.subscription
        """
        if self._subscription is not None:
            self._subscription._unsubscribe()
        self._subscription = None
        self._subscribed_pattern = None
        self._listener = None
        self._subscription_type = SubscriptionType.NONE

    def subscribe_from_pattern(self, topics: Set[str]):
        """ Change subscription on cluster metadata update if a new topic
        created or one is removed.

        Caller: Coordinator
        Affects: SubscriptionState.subscription
        """
        assert self._subscription_type == SubscriptionType.AUTO_PATTERN
        self._change_subscription(Subscription(topics))

    def assign_from_subscribed(self, assignment: Set[TopicPartition]):
        """ Set assignment if automatic assignment is used.

        Caller: Coordinator
        Affects: SubscriptionState.subscription.assignment
        """
        assert self._subscription_type in [SubscriptionType.AUTO_PATTERN, SubscriptionType.AUTO_TOPICS]
        self._subscription._assign(assignment)
        self._notify_assignment_waiters()

    def begin_reassignment(self):
        """ Signal from Coordinator that a group re-join is needed. For example
        this will be called if a commit or heartbeat fails with an
        InvalidMember error.

        Caller: Coordinator
        """
        if self._subscription is not None:
            self._subscription._begin_reassignment()

    def seek(self, tp: TopicPartition, offset: int):
        """ Force reset of position to the specified offset.

        Caller: Consumer, Fetcher
        Affects: TopicPartitionState.position
        """
        self._assigned_state(tp).seek(offset)

    def wait_for_subscription(self):
        """ Wait for subscription change. This will always wait for next
        subscription.
        """
        fut = create_future()
        self._subscription_waiters.append(fut)
        return fut

    def wait_for_assignment(self):
        """ Wait for next assignment. Be careful, as this will always wait for
        next assignment, even if the current one is active.
        """
        fut = create_future()
        self._assignment_waiters.append(fut)
        return fut

    def register_fetch_waiters(self, waiters):
        self._fetch_waiters = waiters

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

    def pause(self, tp: TopicPartition) -> None:
        self._assigned_state(tp).pause()

    def paused_partitions(self) -> Set[TopicPartition]:
        res = set()
        for tp in self.assigned_partitions():
            if self._assigned_state(tp).paused:
                res.add(tp)
        return res

    def resume(self, tp: TopicPartition) -> None:
        self._assigned_state(tp).resume()

    @contextlib.contextmanager
    def fetch_context(self):
        self._fetch_count += 1
        yield
        self._fetch_count -= 1
        if self._fetch_count == 0:
            self._last_fetch_ended = time.monotonic()

    @property
    def fetcher_idle_time(self):
        """ How much time (in seconds) spent without consuming any records """
        if self._fetch_count == 0:
            return time.monotonic() - self._last_fetch_ended
        else:
            return 0