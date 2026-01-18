import asyncio
import logging
import re
import sys
import traceback
import warnings
from typing import Dict, List
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.client import AIOKafkaClient
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.errors import (
from aiokafka.structs import TopicPartition, ConsumerRecord
from aiokafka.util import (
from aiokafka import __version__
from .fetcher import Fetcher, OffsetResetStrategy
from .group_coordinator import GroupCoordinator, NoGroupCoordinator
from .subscription_state import SubscriptionState
def _validate_topics(self, topics):
    if not isinstance(topics, (tuple, set, list)):
        raise ValueError('Topics should be list of strings')
    return set(topics)