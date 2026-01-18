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
class StickyAssignmentExecutor:

    def __init__(self, cluster, members):
        self.members = members
        self.current_assignment = defaultdict(list)
        self.previous_assignment = {}
        self.current_partition_consumer = {}
        self.is_fresh_assignment = False
        self.partition_to_all_potential_consumers = {}
        self.consumer_to_all_potential_partitions = {}
        self.sorted_current_subscriptions = SortedSet()
        self.sorted_partitions = []
        self.unassigned_partitions = []
        self.revocation_required = False
        self.partition_movements = PartitionMovements()
        self._initialize(cluster)

    def perform_initial_assignment(self):
        self._populate_sorted_partitions()
        self._populate_partitions_to_reassign()

    def balance(self):
        self._initialize_current_subscriptions()
        initializing = len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) == 0
        for partition in self.unassigned_partitions:
            if not self.partition_to_all_potential_consumers[partition]:
                continue
            self._assign_partition(partition)
        fixed_partitions = set()
        for partition in self.partition_to_all_potential_consumers.keys():
            if not self._can_partition_participate_in_reassignment(partition):
                fixed_partitions.add(partition)
        for fixed_partition in fixed_partitions:
            remove_if_present(self.sorted_partitions, fixed_partition)
            remove_if_present(self.unassigned_partitions, fixed_partition)
        fixed_assignments = {}
        for consumer in self.consumer_to_all_potential_partitions.keys():
            if not self._can_consumer_participate_in_reassignment(consumer):
                self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
                fixed_assignments[consumer] = self.current_assignment[consumer]
                del self.current_assignment[consumer]
        prebalance_assignment = deepcopy(self.current_assignment)
        prebalance_partition_consumers = deepcopy(self.current_partition_consumer)
        if not self.revocation_required:
            self._perform_reassignments(self.unassigned_partitions)
        reassignment_performed = self._perform_reassignments(self.sorted_partitions)
        if not initializing and reassignment_performed and (self._get_balance_score(self.current_assignment) >= self._get_balance_score(prebalance_assignment)):
            self.current_assignment = prebalance_assignment
            self.current_partition_consumer.clear()
            self.current_partition_consumer.update(prebalance_partition_consumers)
        for consumer, partitions in fixed_assignments.items():
            self.current_assignment[consumer] = partitions
            self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)

    def get_final_assignment(self, member_id):
        assignment = defaultdict(list)
        for topic_partition in self.current_assignment[member_id]:
            assignment[topic_partition.topic].append(topic_partition.partition)
        assignment = {k: sorted(v) for k, v in assignment.items()}
        return assignment.items()

    def _initialize(self, cluster):
        self._init_current_assignments(self.members)
        for topic in cluster.topics():
            partitions = cluster.partitions_for_topic(topic)
            if partitions is None:
                log.warning('No partition metadata for topic %s', topic)
                continue
            for p in partitions:
                partition = TopicPartition(topic=topic, partition=p)
                self.partition_to_all_potential_consumers[partition] = []
        for consumer_id, member_metadata in self.members.items():
            self.consumer_to_all_potential_partitions[consumer_id] = []
            for topic in member_metadata.subscription:
                if cluster.partitions_for_topic(topic) is None:
                    log.warning('No partition metadata for topic {}'.format(topic))
                    continue
                for p in cluster.partitions_for_topic(topic):
                    partition = TopicPartition(topic=topic, partition=p)
                    self.consumer_to_all_potential_partitions[consumer_id].append(partition)
                    self.partition_to_all_potential_consumers[partition].append(consumer_id)
            if consumer_id not in self.current_assignment:
                self.current_assignment[consumer_id] = []

    def _init_current_assignments(self, members):
        sorted_partition_consumers_by_generation = {}
        for consumer, member_metadata in members.items():
            for partitions in member_metadata.partitions:
                if partitions in sorted_partition_consumers_by_generation:
                    consumers = sorted_partition_consumers_by_generation[partitions]
                    if member_metadata.generation and member_metadata.generation in consumers:
                        log.warning('Partition {} is assigned to multiple consumers following sticky assignment generation {}.'.format(partitions, member_metadata.generation))
                    else:
                        consumers[member_metadata.generation] = consumer
                else:
                    sorted_consumers = {member_metadata.generation: consumer}
                    sorted_partition_consumers_by_generation[partitions] = sorted_consumers
        for partitions, consumers in sorted_partition_consumers_by_generation.items():
            generations = sorted(consumers.keys(), reverse=True)
            self.current_assignment[consumers[generations[0]]].append(partitions)
            if len(generations) > 1:
                self.previous_assignment[partitions] = ConsumerGenerationPair(consumer=consumers[generations[1]], generation=generations[1])
        self.is_fresh_assignment = len(self.current_assignment) == 0
        for consumer_id, partitions in self.current_assignment.items():
            for partition in partitions:
                self.current_partition_consumer[partition] = consumer_id

    def _are_subscriptions_identical(self):
        """
        Returns:
            true, if both potential consumers of partitions and potential partitions
            that consumers can consume are the same
        """
        if not has_identical_list_elements(list(self.partition_to_all_potential_consumers.values())):
            return False
        return has_identical_list_elements(list(self.consumer_to_all_potential_partitions.values()))

    def _populate_sorted_partitions(self):
        all_partitions = set(((tp, tuple(consumers)) for tp, consumers in self.partition_to_all_potential_consumers.items()))
        partitions_sorted_by_num_of_potential_consumers = sorted(all_partitions, key=partitions_comparator_key)
        self.sorted_partitions = []
        if not self.is_fresh_assignment and self._are_subscriptions_identical():
            assignments = deepcopy(self.current_assignment)
            for partitions in assignments.values():
                to_remove = []
                for partition in partitions:
                    if partition not in self.partition_to_all_potential_consumers:
                        to_remove.append(partition)
                for partition in to_remove:
                    partitions.remove(partition)
            sorted_consumers = SortedSet(iterable=[(consumer, tuple(partitions)) for consumer, partitions in assignments.items()], key=subscriptions_comparator_key)
            while sorted_consumers:
                consumer, _ = sorted_consumers.pop_last()
                remaining_partitions = assignments[consumer]
                previous_partitions = set(self.previous_assignment.keys()).intersection(set(remaining_partitions))
                if previous_partitions:
                    partition = previous_partitions.pop()
                    remaining_partitions.remove(partition)
                    self.sorted_partitions.append(partition)
                    sorted_consumers.add((consumer, tuple(assignments[consumer])))
                elif remaining_partitions:
                    self.sorted_partitions.append(remaining_partitions.pop())
                    sorted_consumers.add((consumer, tuple(assignments[consumer])))
            while partitions_sorted_by_num_of_potential_consumers:
                partition = partitions_sorted_by_num_of_potential_consumers.pop(0)[0]
                if partition not in self.sorted_partitions:
                    self.sorted_partitions.append(partition)
        else:
            while partitions_sorted_by_num_of_potential_consumers:
                self.sorted_partitions.append(partitions_sorted_by_num_of_potential_consumers.pop(0)[0])

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

    def _initialize_current_subscriptions(self):
        self.sorted_current_subscriptions = SortedSet(iterable=[(consumer, tuple(partitions)) for consumer, partitions in self.current_assignment.items()], key=subscriptions_comparator_key)

    def _get_consumer_with_least_subscriptions(self):
        return self.sorted_current_subscriptions.first()[0]

    def _get_consumer_with_most_subscriptions(self):
        return self.sorted_current_subscriptions.last()[0]

    def _remove_consumer_from_current_subscriptions_and_maintain_order(self, consumer):
        self.sorted_current_subscriptions.remove((consumer, tuple(self.current_assignment[consumer])))

    def _add_consumer_to_current_subscriptions_and_maintain_order(self, consumer):
        self.sorted_current_subscriptions.add((consumer, tuple(self.current_assignment[consumer])))

    def _is_balanced(self):
        """Determines if the current assignment is a balanced one"""
        if len(self.current_assignment[self._get_consumer_with_least_subscriptions()]) >= len(self.current_assignment[self._get_consumer_with_most_subscriptions()]) - 1:
            return True
        all_assigned_partitions = {}
        for consumer_id, consumer_partitions in self.current_assignment.items():
            for partition in consumer_partitions:
                if partition in all_assigned_partitions:
                    log.error('{} is assigned to more than one consumer.'.format(partition))
                all_assigned_partitions[partition] = consumer_id
        for consumer, _ in self.sorted_current_subscriptions:
            consumer_partition_count = len(self.current_assignment[consumer])
            if consumer_partition_count == len(self.consumer_to_all_potential_partitions[consumer]):
                continue
            for partition in self.consumer_to_all_potential_partitions[consumer]:
                if partition not in self.current_assignment[consumer]:
                    other_consumer = all_assigned_partitions[partition]
                    other_consumer_partition_count = len(self.current_assignment[other_consumer])
                    if consumer_partition_count < other_consumer_partition_count:
                        return False
        return True

    def _assign_partition(self, partition):
        for consumer, _ in self.sorted_current_subscriptions:
            if partition in self.consumer_to_all_potential_partitions[consumer]:
                self._remove_consumer_from_current_subscriptions_and_maintain_order(consumer)
                self.current_assignment[consumer].append(partition)
                self.current_partition_consumer[partition] = consumer
                self._add_consumer_to_current_subscriptions_and_maintain_order(consumer)
                break

    def _can_partition_participate_in_reassignment(self, partition):
        return len(self.partition_to_all_potential_consumers[partition]) >= 2

    def _can_consumer_participate_in_reassignment(self, consumer):
        current_partitions = self.current_assignment[consumer]
        current_assignment_size = len(current_partitions)
        max_assignment_size = len(self.consumer_to_all_potential_partitions[consumer])
        if current_assignment_size > max_assignment_size:
            log.error('The consumer {} is assigned more partitions than the maximum possible.'.format(consumer))
        if current_assignment_size < max_assignment_size:
            return True
        for partition in current_partitions:
            if self._can_partition_participate_in_reassignment(partition):
                return True
        return False

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

    def _reassign_partition(self, partition):
        new_consumer = None
        for another_consumer, _ in self.sorted_current_subscriptions:
            if partition in self.consumer_to_all_potential_partitions[another_consumer]:
                new_consumer = another_consumer
                break
        assert new_consumer is not None
        self._reassign_partition_to_consumer(partition, new_consumer)

    def _reassign_partition_to_consumer(self, partition, new_consumer):
        consumer = self.current_partition_consumer[partition]
        partition_to_be_moved = self.partition_movements.get_partition_to_be_moved(partition, consumer, new_consumer)
        self._move_partition(partition_to_be_moved, new_consumer)

    def _move_partition(self, partition, new_consumer):
        old_consumer = self.current_partition_consumer[partition]
        self._remove_consumer_from_current_subscriptions_and_maintain_order(old_consumer)
        self._remove_consumer_from_current_subscriptions_and_maintain_order(new_consumer)
        self.partition_movements.move_partition(partition, old_consumer, new_consumer)
        self.current_assignment[old_consumer].remove(partition)
        self.current_assignment[new_consumer].append(partition)
        self.current_partition_consumer[partition] = new_consumer
        self._add_consumer_to_current_subscriptions_and_maintain_order(new_consumer)
        self._add_consumer_to_current_subscriptions_and_maintain_order(old_consumer)

    @staticmethod
    def _get_balance_score(assignment):
        """Calculates a balance score of a give assignment
        as the sum of assigned partitions size difference of all consumer pairs.
        A perfectly balanced assignment (with all consumers getting the same number of
        partitions) has a balance score of 0. Lower balance score indicates a more
        balanced assignment.

        Arguments:
          assignment (dict): {consumer: list of assigned topic partitions}

        Returns:
          the balance score of the assignment
        """
        score = 0
        consumer_to_assignment = {}
        for consumer_id, partitions in assignment.items():
            consumer_to_assignment[consumer_id] = len(partitions)
        consumers_to_explore = set(consumer_to_assignment.keys())
        for consumer_id in consumer_to_assignment.keys():
            if consumer_id in consumers_to_explore:
                consumers_to_explore.remove(consumer_id)
                for other_consumer_id in consumers_to_explore:
                    score += abs(consumer_to_assignment[consumer_id] - consumer_to_assignment[other_consumer_id])
        return score