import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_and_get_group(group_name):
    """Check the existence and return the group handle."""
    _check_inside_actor()
    global _group_mgr
    if not is_group_initialized(group_name):
        try:
            name = 'info_' + group_name
            mgr = ray.get_actor(name=name)
            ids, world_size, rank, backend = ray.get(mgr.get_info.remote())
            worker = ray._private.worker.global_worker
            id_ = worker.core_worker.get_actor_id()
            r = rank[ids.index(id_)]
            _group_mgr.create_collective_group(backend, world_size, r, group_name)
        except ValueError as exc:
            if 'collective_group_name' in os.environ and os.environ['collective_group_name'] == group_name:
                rank = int(os.environ['collective_rank'])
                world_size = int(os.environ['collective_world_size'])
                backend = os.environ['collective_backend']
                _group_mgr.create_collective_group(backend, world_size, rank, group_name)
            else:
                raise RuntimeError("The collective group '{}' is not initialized in the process.".format(group_name)) from exc
    g = _group_mgr.get_group_by_name(group_name)
    return g