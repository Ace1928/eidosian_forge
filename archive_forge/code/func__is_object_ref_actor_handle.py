import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def _is_object_ref_actor_handle(self) -> bool:
    object_ref_hex = self.object_ref.hex()
    taskid_random_bits_size = (TASKID_BYTES_SIZE - ACTORID_BYTES_SIZE) * 2
    actorid_random_bits_size = (ACTORID_BYTES_SIZE - JOBID_BYTES_SIZE) * 2
    random_bits = object_ref_hex[:taskid_random_bits_size]
    actor_random_bits = object_ref_hex[taskid_random_bits_size:taskid_random_bits_size + actorid_random_bits_size]
    if random_bits == 'f' * 16 and (not actor_random_bits == 'f' * 24):
        return True
    else:
        return False