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
def _assign_partition(self, partition):
    for consumer, _ in self.sorted_current_subscriptions:
        if partition in self.consumer_to_all_potential_partitions[consumer]:
            self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
            self.current_assignment[consumer].append(partition)
            self.current_partition_consumer[partition] = consumer
            self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)
            break