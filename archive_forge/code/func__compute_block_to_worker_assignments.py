import bisect
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
import ray
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockAccessor
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def _compute_block_to_worker_assignments(self):
    block_to_workers: dict[int, List['ray.ActorHandle']] = defaultdict(list)
    worker_to_blocks: dict['ray.ActorHandle', List[int]] = defaultdict(list)
    loc_to_workers: dict[str, List['ray.ActorHandle']] = defaultdict(list)
    locs = ray.get([w.ping.remote() for w in self._workers])
    for i, loc in enumerate(locs):
        loc_to_workers[loc].append(self._workers[i])
    block_locs = ray.experimental.get_object_locations(self._non_empty_blocks)
    for block_idx, block in enumerate(self._non_empty_blocks):
        block_info = block_locs[block]
        locs = block_info.get('node_ids', [])
        for loc in locs:
            for worker in loc_to_workers[loc]:
                block_to_workers[block_idx].append(worker)
                worker_to_blocks[worker].append(block_idx)
    for block_idx, block in enumerate(self._non_empty_blocks):
        if len(block_to_workers[block_idx]) == 0:
            worker = random.choice(self._workers)
            block_to_workers[block_idx].append(worker)
            worker_to_blocks[worker].append(block_idx)
    return (block_to_workers, worker_to_blocks)