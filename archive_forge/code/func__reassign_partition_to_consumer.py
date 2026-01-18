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
def _reassign_partition_to_consumer(self, partition, new_consumer):
    consumer = self.current_partition_consumer[partition]
    partition_to_be_moved = self.partition_movements.get_partition_to_be_moved(partition, consumer, new_consumer)
    self._move_partition(partition_to_be_moved, new_consumer)