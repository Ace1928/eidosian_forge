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
class FetchError:

    def __init__(self, *, error, backoff):
        self._error = error
        self._created = time.monotonic()
        self._backoff = backoff

    def calculate_backoff(self):
        lifetime = time.monotonic() - self._created
        if lifetime < self._backoff:
            return self._backoff - lifetime
        return 0

    def check_raise(self):
        raise self._error

    def __repr__(self):
        return f'<FetchError error={self._error!r}>'