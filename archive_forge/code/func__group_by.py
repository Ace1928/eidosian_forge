import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def _group_by(self, group_by_type: GroupByType):
    """Group entries and summarize the result.

        NOTE: Each group is another MemoryTable.
        """
    self.group = {}
    group = defaultdict(list)
    for entry in self.table:
        group[entry.group_key(group_by_type)].append(entry)
    for group_key, entries in group.items():
        self.group[group_key] = MemoryTable(entries, group_by_type=None, sort_by_type=None)
    for group_key, group_memory_table in self.group.items():
        group_memory_table.summarize()
    return self