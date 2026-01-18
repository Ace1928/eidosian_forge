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
class TransactionContext:

    def __init__(self, producer):
        self._producer = producer

    async def __aenter__(self):
        await self._producer.begin_transaction()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if self._producer._txn_manager.is_fatal_error():
                return
            await self._producer.abort_transaction()
        else:
            await self._producer.commit_transaction()