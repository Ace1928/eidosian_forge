import asyncio
import logging
import sys
import traceback
import warnings
from aiokafka.client import AIOKafkaClient
from aiokafka.codec import has_gzip, has_snappy, has_lz4, has_zstd
from aiokafka.errors import (
from aiokafka.partitioner import DefaultPartitioner
from aiokafka.record.default_records import DefaultRecordBatch
from aiokafka.record.legacy_records import LegacyRecordBatchBuilder
from aiokafka.structs import TopicPartition
from aiokafka.util import (
from .message_accumulator import MessageAccumulator
from .sender import Sender
from .transaction_manager import TransactionManager
def _ensure_transactional(self):
    if self._txn_manager is None or self._txn_manager.transactional_id is None:
        raise IllegalOperation('You need to configure transaction_id to use transactions')