import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def _remove_movement_record_of_partition(self, partition):
    pair = self.partition_movements[partition]
    del self.partition_movements[partition]
    self.partition_movements_by_topic[partition.topic][pair].remove(partition)
    if not self.partition_movements_by_topic[partition.topic][pair]:
        del self.partition_movements_by_topic[partition.topic][pair]
    if not self.partition_movements_by_topic[partition.topic]:
        del self.partition_movements_by_topic[partition.topic]
    return pair