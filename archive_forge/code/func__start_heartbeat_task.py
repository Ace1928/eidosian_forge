import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
def _start_heartbeat_task(self):
    if self._heartbeat_task is None:
        self._heartbeat_task = create_task(self._heartbeat_routine())