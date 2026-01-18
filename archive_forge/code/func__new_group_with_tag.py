import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def _new_group_with_tag(ranks=None, timeout=None, backend=None, pg_options=None, pg_tag=None, use_local_synchronization=False):
    """
    Variant of ``new_group`` that exposes tag creation.

    :: N.B. The mechanism is experimental and tied to the functional collectives effort, see
    ``torch.distributed._functional_collectives`` for reference on how to use it.
    """
    global _world
    default_pg = _get_default_group()
    default_backend, default_store = _world.pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()
    if not backend:
        backend = default_backend
    backend = Backend(backend)
    if timeout is None:
        timeout = _get_default_timeout(backend)
    _check_valid_timeout(timeout)
    if use_local_synchronization:
        if backend == Backend.MPI:
            raise ValueError("MPI backend doesn't support use_local_synchronization=True")
        if ranks is not None and get_rank() not in ranks:
            return None
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise ValueError("the new group's world size should be less or equal to the world size set by init_process_group")
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise ValueError("The new group's rank should be within the world_size set by init_process_group")
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank
    group_name = _process_group_name(ranks, use_hashed_name=use_local_synchronization)
    pg, pg_store = _new_process_group_helper(group_world_size, group_rank, ranks, backend, default_store, group_name, pg_options=pg_options, timeout=timeout, pg_tag=pg_tag)
    _world.pg_group_ranks[pg] = {global_rank: group_rank for group_rank, global_rank in enumerate(ranks)}
    if _is_barrier_after_init() == 1:
        logger.info('Performing barrier after ProcessGroup initialization since TORCH_DIST_INIT_BARRIER = 1')
        if backend == Backend.MPI:
            barrier()
        else:
            barrier_store = pg_store if use_local_synchronization else default_store
            world_size = len(ranks) if use_local_synchronization else get_world_size()
            _store_based_barrier(global_rank, barrier_store, group_name, world_size, timeout)
    return pg