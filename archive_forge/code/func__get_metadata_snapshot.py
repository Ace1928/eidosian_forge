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
def _get_metadata_snapshot(self):
    partitions_per_topic = {}
    for topic in self._group_subscription:
        partitions = self._cluster.partitions_for_topic(topic) or []
        partitions_per_topic[topic] = len(partitions)
    return partitions_per_topic