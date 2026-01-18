import asyncio
import collections
import copy
import time
from aiokafka.errors import (KafkaTimeoutError,
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.record.default_records import DefaultRecordBatchBuilder
from aiokafka.structs import RecordMetadata
from aiokafka.util import create_future, get_running_loop
def _set_producer_state(self, producer_id, producer_epoch, base_sequence):
    assert type(self._builder) is DefaultRecordBatchBuilder
    self._builder.set_producer_state(producer_id, producer_epoch, base_sequence)