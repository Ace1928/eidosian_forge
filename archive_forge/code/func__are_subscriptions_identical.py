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
def _are_subscriptions_identical(self):
    """
        Returns:
            true, if both potential consumers of partitions and potential partitions
            that consumers can consume are the same
        """
    if not has_identical_list_elements(list(self.partition_to_all_potential_consumers.values())):
        return False
    return has_identical_list_elements(list(self.consumer_to_all_potential_partitions.values()))