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