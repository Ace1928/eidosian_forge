import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def _add_partition_movement_record(self, partition, pair):
    self.partition_movements[partition] = pair
    self.partition_movements_by_topic[partition.topic][pair].add(partition)