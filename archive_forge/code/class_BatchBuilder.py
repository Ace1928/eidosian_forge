import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
class BatchBuilder:

    def __init__(self, magic, batch_size, compression_type, *, is_transactional):
        if magic < 2:
            assert not is_transactional
            self._builder = LegacyRecordBatchBuilder(magic, compression_type, batch_size)
        else:
            self._builder = DefaultRecordBatchBuilder(magic, compression_type, is_transactional=is_transactional, producer_id=-1, producer_epoch=-1, base_sequence=0, batch_size=batch_size)
        self._relative_offset = 0
        self._buffer = None
        self._closed = False

    def append(self, *, timestamp, key, value, headers=[]):
        """Add a message to the batch.

        Arguments:
            timestamp (float or None): epoch timestamp in seconds. If None,
                the timestamp will be set to the current time. If submitting to
                an 0.8.x or 0.9.x broker, the timestamp will be ignored.
            key (bytes or None): the message key. `key` and `value` may not
                both be None.
            value (bytes or None): the message value. `key` and `value` may not
                both be None.

        Returns:
            If the message was successfully added, returns a metadata object
            with crc, offset, size, and timestamp fields. If the batch is full
            or closed, returns None.
        """
        if self._closed:
            return None
        metadata = self._builder.append(self._relative_offset, timestamp, key, value, headers=headers)
        if metadata is None:
            return None
        self._relative_offset += 1
        return metadata

    def close(self):
        """Close the batch to further updates.

        Closing the batch before submitting to the producer ensures that no
        messages are added via the ``producer.send()`` interface. To gracefully
        support both the batch and individual message interfaces, leave the
        batch open. For complete control over the batch's contents, close
        before submission. Closing a batch has no effect on when it's sent to
        the broker.

        A batch may not be reopened after it's closed.
        """
        if self._closed:
            return
        self._closed = True

    def _set_producer_state(self, producer_id, producer_epoch, base_sequence):
        assert type(self._builder) is DefaultRecordBatchBuilder
        self._builder.set_producer_state(producer_id, producer_epoch, base_sequence)

    def _build(self):
        self.close()
        if self._buffer is None:
            self._buffer = self._builder.build()
            del self._builder
        return self._buffer

    def size(self):
        """Get the size of batch in bytes."""
        if self._buffer is not None:
            return len(self._buffer)
        else:
            return self._builder.size()

    def record_count(self):
        """Get the number of records in the batch."""
        return self._relative_offset