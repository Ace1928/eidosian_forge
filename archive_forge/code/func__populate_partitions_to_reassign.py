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
def _populate_partitions_to_reassign(self):
    self.unassigned_partitions = deepcopy(self.sorted_partitions)
    assignments_to_remove = []
    for consumer_id, partitions in self.current_assignment.items():
        if consumer_id not in self.members:
            for partition in partitions:
                del self.current_partition_consumer[partition]
            assignments_to_remove.append(consumer_id)
        else:
            partitions_to_remove = []
            for partition in partitions:
                if partition not in self.partition_to_all_potential_consumers:
                    partitions_to_remove.append(partition)
                elif partition.topic not in self.members[consumer_id].subscription:
                    partitions_to_remove.append(partition)
                    self.revocation_required = True
                else:
                    self.unassigned_partitions.remove(partition)
            for partition in partitions_to_remove:
                self.current_assignment[consumer_id].remove(partition)
                del self.current_partition_consumer[partition]
    for consumer_id in assignments_to_remove:
        del self.current_assignment[consumer_id]