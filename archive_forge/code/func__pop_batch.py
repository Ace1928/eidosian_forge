import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def _pop_batch(self, tp):
    batch = self._batches[tp].popleft()
    not_retry = batch.retry_count == 0
    if self._txn_manager is not None and not_retry:
        assert self._txn_manager.has_pid(), 'We should have waited for it in sender routine'
        seq = self._txn_manager.sequence_number(batch.tp)
        self._txn_manager.increment_sequence_number(batch.tp, batch.record_count)
        batch.set_producer_state(producer_id=self._txn_manager.producer_id, producer_epoch=self._txn_manager.producer_epoch, base_sequence=seq)
    batch.drain_ready()
    if len(self._batches[tp]) == 0:
        del self._batches[tp]
    self._pending_batches.add(batch)
    if not_retry:

        def cb(fut, batch=batch, self=self):
            self._pending_batches.remove(batch)
        batch.future.add_done_callback(cb)
    return batch