import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
class StickyAssignorUserDataV1(Struct):
    """
    Used for preserving consumer's previously assigned partitions
    list and sending it as user data to the leader during a rebalance
    """
    SCHEMA = Schema(('previous_assignment', Array(('topic', String('utf-8')), ('partitions', Array(Int32)))), ('generation', Int32))