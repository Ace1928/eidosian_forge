import asyncio
import collections
import logging
import random
import time
from itertools import chain
import async_timeout
import aiokafka.errors as Errors
from aiokafka.errors import (
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.record.memory_records import MemoryRecords
from aiokafka.record.control_record import ControlRecord, ABORT_MARKER
from aiokafka.structs import OffsetAndTimestamp, TopicPartition, ConsumerRecord
from aiokafka.util import create_future, create_task
class PartitionRecords:

    def __init__(self, tp, records, aborted_transactions, fetch_offset, key_deserializer, value_deserializer, check_crcs, isolation_level):
        self._tp = tp
        self._records = records
        self._aborted_transactions = sorted(aborted_transactions or [], key=lambda x: x[1])
        self._aborted_producers = set()
        self._key_deserializer = key_deserializer
        self._value_deserializer = value_deserializer
        self._check_crcs = check_crcs
        self._isolation_level = isolation_level
        self.next_fetch_offset = fetch_offset
        self._records_iterator = self._unpack_records()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._records_iterator)
        except StopIteration:
            self._records_iterator = None
            raise

    def _unpack_records(self):
        tp = self._tp
        records = self._records
        while records.has_next():
            next_batch = records.next_batch()
            if self._check_crcs and (not next_batch.validate_crc()):
                raise Errors.CorruptRecordException(f'Invalid CRC - {tp}')
            if self._isolation_level == READ_COMMITTED and next_batch.producer_id is not None:
                self._consume_aborted_up_to(next_batch.base_offset)
                if next_batch.is_control_batch:
                    if self._contains_abort_marker(next_batch):
                        self._aborted_producers.discard(next_batch.producer_id)
                if next_batch.is_transactional and next_batch.producer_id in self._aborted_producers:
                    log.debug('Skipping aborted record batch from partition %s with producer_id %s and offsets %s to %s', tp, next_batch.producer_id, next_batch.base_offset, next_batch.next_offset - 1)
                    self.next_fetch_offset = next_batch.next_offset
                    continue
            if next_batch.is_control_batch:
                self.next_fetch_offset = next_batch.next_offset
                continue
            for record in next_batch:
                if record.offset < self.next_fetch_offset:
                    continue
                consumer_record = self._consumer_record(tp, record)
                self.next_fetch_offset = record.offset + 1
                yield consumer_record
            self.next_fetch_offset = next_batch.next_offset

    def _consume_aborted_up_to(self, batch_offset):
        aborted_transactions = self._aborted_transactions
        while aborted_transactions:
            producer_id, first_offset = aborted_transactions[0]
            if first_offset <= batch_offset:
                self._aborted_producers.add(producer_id)
                aborted_transactions.pop(0)
            else:
                break

    def _contains_abort_marker(self, next_batch):
        try:
            control_record = next(next_batch)
        except StopIteration:
            raise Errors.KafkaError('Control batch did not contain any records')
        return ControlRecord.parse(control_record.key) == ABORT_MARKER

    def _consumer_record(self, tp, record):
        key_size = len(record.key) if record.key is not None else -1
        value_size = len(record.value) if record.value is not None else -1
        if self._key_deserializer:
            key = self._key_deserializer(record.key)
        else:
            key = record.key
        if self._value_deserializer:
            value = self._value_deserializer(record.value)
        else:
            value = record.value
        return ConsumerRecord(tp.topic, tp.partition, record.offset, record.timestamp, record.timestamp_type, key, value, record.checksum, key_size, value_size, tuple(record.headers))