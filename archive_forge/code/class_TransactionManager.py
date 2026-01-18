from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
class TransactionManager:

    def __init__(self, transactional_id, transaction_timeout_ms):
        self.transactional_id = transactional_id
        self.transaction_timeout_ms = transaction_timeout_ms
        self.state = TransactionState.UNINITIALIZED
        self._pid_and_epoch = PidAndEpoch(NO_PRODUCER_ID, NO_PRODUCER_EPOCH)
        self._pid_waiter = create_future()
        self._sequence_numbers = defaultdict(lambda: 0)
        self._transaction_waiter = None
        self._task_waiter = None
        self._txn_partitions = set()
        self._pending_txn_partitions = set()
        self._txn_consumer_group = None
        self._pending_txn_offsets = deque()

    def set_pid_and_epoch(self, pid: int, epoch: int):
        self._pid_and_epoch = PidAndEpoch(pid, epoch)
        self._pid_waiter.set_result(None)
        if self.transactional_id:
            self._transition_to(TransactionState.READY)

    def has_pid(self):
        return self._pid_and_epoch.pid != NO_PRODUCER_ID

    async def wait_for_pid(self):
        if self.has_pid():
            return
        else:
            await self._pid_waiter

    def sequence_number(self, tp: TopicPartition):
        return self._sequence_numbers[tp]

    def increment_sequence_number(self, tp: TopicPartition, increment: int):
        seq = self._sequence_numbers[tp] + increment
        if seq > 2 ** 31 - 1:
            seq -= 2 ** 32
        self._sequence_numbers[tp] = seq

    @property
    def producer_id(self):
        return self._pid_and_epoch.pid

    @property
    def producer_epoch(self):
        return self._pid_and_epoch.epoch

    def _transition_to(self, target):
        assert TransactionState.is_transition_valid(self.state, target), f'Invalid state transition {self.state} -> {target}'
        self.state = target

    def begin_transaction(self):
        self._transition_to(TransactionState.IN_TRANSACTION)
        self._transaction_waiter = create_future()

    def committing_transaction(self):
        if self.state == TransactionState.ABORTABLE_ERROR:
            self._transaction_waiter.result()
        self._transition_to(TransactionState.COMMITTING_TRANSACTION)
        self.notify_task_waiter()

    def aborting_transaction(self):
        self._transition_to(TransactionState.ABORTING_TRANSACTION)
        if self._transaction_waiter.done():
            self._transaction_waiter = create_future()
        self.notify_task_waiter()

    def complete_transaction(self):
        assert not self._pending_txn_partitions
        assert not self._pending_txn_offsets
        self._transition_to(TransactionState.READY)
        self._txn_partitions.clear()
        self._txn_consumer_group = None
        if not self._transaction_waiter.done():
            self._transaction_waiter.set_result(None)

    def error_transaction(self, exc):
        self._transition_to(TransactionState.ABORTABLE_ERROR)
        self._txn_partitions.clear()
        self._txn_consumer_group = None
        self._pending_txn_partitions.clear()
        for _, _, fut in self._pending_txn_offsets:
            fut.set_exception(exc)
        self._pending_txn_offsets.clear()
        self._transaction_waiter.set_exception(exc)

    def fatal_error(self, exc):
        self._transition_to(TransactionState.FATAL_ERROR)
        self._txn_partitions.clear()
        self._txn_consumer_group = None
        self._pending_txn_partitions.clear()
        for _, _, fut in self._pending_txn_offsets:
            fut.set_exception(exc)
        self._pending_txn_offsets.clear()
        if self._transaction_waiter.done():
            self._transaction_waiter = create_future()
        self._transaction_waiter.set_exception(exc)

    def maybe_add_partition_to_txn(self, tp: TopicPartition):
        if self.transactional_id is None:
            return
        assert self.is_in_transaction()
        if tp not in self._txn_partitions:
            self._pending_txn_partitions.add(tp)
            self.notify_task_waiter()

    def add_offsets_to_txn(self, offsets, group_id):
        assert self.is_in_transaction()
        assert self.transactional_id
        fut = create_future()
        self._pending_txn_offsets.append((group_id, offsets, fut))
        self.notify_task_waiter()
        return fut

    def is_in_transaction(self):
        return self.state == TransactionState.IN_TRANSACTION

    def partitions_to_add(self):
        return self._pending_txn_partitions

    def consumer_group_to_add(self):
        if self._txn_consumer_group is not None:
            return
        for group_id, _, _ in self._pending_txn_offsets:
            return group_id

    def offsets_to_commit(self):
        if self._txn_consumer_group is None:
            return
        for group_id, offsets, _ in self._pending_txn_offsets:
            return (offsets, group_id)

    def partition_added(self, tp: TopicPartition):
        self._pending_txn_partitions.remove(tp)
        self._txn_partitions.add(tp)

    def consumer_group_added(self, group_id):
        self._txn_consumer_group = group_id

    def offset_committed(self, tp, offset, group_id):
        pending_group_id, pending_offsets, fut = self._pending_txn_offsets[0]
        assert pending_group_id == group_id
        assert tp in pending_offsets and pending_offsets[tp].offset == offset
        del pending_offsets[tp]
        if not pending_offsets:
            fut.set_result(None)
            self._pending_txn_offsets.popleft()

    @property
    def txn_partitions(self):
        return self._txn_partitions

    def needs_transaction_commit(self):
        if self.state == TransactionState.COMMITTING_TRANSACTION:
            return TransactionResult.COMMIT
        elif self.state == TransactionState.ABORTING_TRANSACTION:
            return TransactionResult.ABORT
        else:
            return

    def is_empty_transaction(self):
        return len(self.txn_partitions) == 0 and self._txn_consumer_group is None

    def is_fatal_error(self):
        return self.state == TransactionState.FATAL_ERROR

    def wait_for_transaction_end(self):
        return self._transaction_waiter

    def notify_task_waiter(self):
        if self._task_waiter is not None and (not self._task_waiter.done()):
            self._task_waiter.set_result(None)

    def make_task_waiter(self):
        self._task_waiter = create_future()
        return self._task_waiter