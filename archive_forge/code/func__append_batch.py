import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def _append_batch(self, builder, tp):
    if self._txn_manager is not None:
        self._txn_manager.maybe_add_partition_to_txn(tp)
    batch = MessageBatch(tp, builder, self._batch_ttl)
    self._batches[tp].append(batch)
    if not self._wait_data_future.done():
        self._wait_data_future.set_result(None)
    return batch