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
 Force a position change to specific offset. Called from
        `Consumer.seek()` API.
        