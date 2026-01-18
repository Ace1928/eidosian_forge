import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def _create_rank_world_size_mappings(self) -> List[Dict]:
    """Create rank and world size mappings for workers.
        There are three maps returned:
            - local_rank_map, which maps from worker world_rank to local_rank.
            - local_world_size_map, which maps from world_rank to local_world_size
            - node_rank_map, which maps from world rank to node rank

        Example:
            Worker 0: 0.0.0.0
            Worker 1: 0.0.0.0
            Worker 2: 0.0.0.1
            Worker 3: 0.0.0.0
            Worker 4: 0.0.0.1

            Workers 0, 1, 3 are on 0.0.0.0.
            Workers 2, 4 are on 0.0.0.1.

            Expected local_rank_map:
            {
                0 -> 0,
                1 -> 1,
                2 -> 0,
                3 -> 2,
                4 -> 1
            }

            Expected local_world_size_map:
            {
                0 -> 3,
                1 -> 3,
                2 -> 2,
                3 -> 3,
                4 -> 2
            }

            Expected node_rank_map:
            {
                0 -> 0,
                1 -> 0,
                2 -> 1,
                3 -> 0,
                4 -> 1
            }

        """
    local_rank_map = {}
    local_world_size_map = {}
    node_rank_map = {}
    node_ips = {}
    node_cnt = 0
    ip_dict = defaultdict(int)
    for world_rank in range(len(self.worker_group)):
        worker = self.worker_group.workers[world_rank]
        node_ip = worker.metadata.node_ip
        local_rank_map[world_rank] = ip_dict[node_ip]
        ip_dict[node_ip] += 1
        if node_ip not in node_ips:
            node_ips[node_ip] = node_cnt
            node_cnt += 1
        node_rank_map[world_rank] = node_ips[node_ip]
    for world_rank in range(len(self.worker_group)):
        worker = self.worker_group.workers[world_rank]
        node_ip = worker.metadata.node_ip
        local_world_size_map[world_rank] = ip_dict[node_ip]
    workers_info = '\n'.join([f'- (ip={w.metadata.node_ip}, pid={w.metadata.pid}) world_rank={i}, local_rank={local_rank_map[i]}, node_rank={node_rank_map[i]}' for i, w in enumerate(self.worker_group.workers)])
    logger.info(f'Started distributed worker processes: \n{workers_info}')
    return (local_rank_map, local_world_size_map, node_rank_map)