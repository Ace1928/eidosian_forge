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
def _perform_reassignments(self, reassignable_partitions):
    reassignment_performed = False
    while True:
        modified = False
        for partition in reassignable_partitions:
            if self._is_balanced():
                break
            if len(self.partition_to_all_potential_consumers[partition]) <= 1:
                log.error('Expected more than one potential consumer for partition {}'.format(partition))
            consumer = self.current_partition_consumer.get(partition)
            if consumer is None:
                log.error('Expected partition {} to be assigned to a consumer'.format(partition))
            if partition in self.previous_assignment and len(self.current_assignment[consumer]) > len(self.current_assignment[self.previous_assignment[partition].consumer]) + 1:
                self._reassign_partition_to_consumer(partition, self.previous_assignment[partition].consumer)
                reassignment_performed = True
                modified = True
                continue
            for other_consumer in self.partition_to_all_potential_consumers[partition]:
                if len(self.current_assignment[consumer]) > len(self.current_assignment[other_consumer]) + 1:
                    self._reassign_partition(partition)
                    reassignment_performed = True
                    modified = True
                    break
        if not modified:
            break
    return reassignment_performed