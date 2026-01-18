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
def _new_process_group_helper(group_size, group_rank, global_ranks_in_group, backend, store, group_name, pg_options=None, timeout=None, pg_tag=None):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world
    if group_name in _world.pg_names.values():
        raise ValueError('The specified group name has already been created, please use a different group name')
    _check_valid_timeout(timeout)
    if pg_tag not in [None, '']:
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            _, prefix_store = _world.pg_map[existing_group]
            return (existing_group, prefix_store)
    is_default_group = len(global_ranks_in_group) == 0
    if not is_default_group:
        global_rank = _get_default_group().rank()
        if global_rank not in global_ranks_in_group:
            return (GroupMember.NON_GROUP_MEMBER, None)
    prefix_store = PrefixStore(f'{group_name}/', store)
    base_pg_options = ProcessGroup.Options(backend=str(backend))
    base_pg_options._timeout = timeout
    pg: ProcessGroup = ProcessGroup(prefix_store, group_rank, group_size, base_pg_options)
    backend_config = BackendConfig(backend)
    for device, backend_str in backend_config.get_device_backend_map().items():
        backend_prefix_store = PrefixStore(f'{device}/', prefix_store)
        if backend_str == Backend.MPI:
            if not is_mpi_available():
                raise RuntimeError("Distributed package doesn't have MPI built in. MPI is only included if you build PyTorch from source on a host that has MPI installed.")
            backend_class = ProcessGroupMPI.create(global_ranks_in_group)
            backend_type = ProcessGroup.BackendType.MPI
            if not backend_class:
                return GroupMember.NON_GROUP_MEMBER
            if pg.rank() == -1 and pg.size() == -1:
                pg = ProcessGroup(backend_prefix_store, backend_class.rank(), backend_class.size(), base_pg_options)
        elif backend_str == Backend.GLOO:
            backend_class = ProcessGroupGloo(backend_prefix_store, group_rank, group_size, timeout=timeout)
            backend_type = ProcessGroup.BackendType.GLOO
        elif backend_str == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL built in")
            if pg_options is not None:
                assert isinstance(pg_options, ProcessGroupNCCL.Options), 'Expected pg_options argument to be of type ProcessGroupNCCL.Options'
                if pg_options._timeout != timeout:
                    warnings.warn('pg_options._timeout was specified, but timeout kwarg has a default value that will always override it. ')
            else:
                pg_options = ProcessGroupNCCL.Options()
                pg_options.is_high_priority_stream = False
            pg_options._timeout = timeout
            split_from = None
            if is_initialized() and _world.default_pg._get_backend_name() == Backend.NCCL and (len(global_ranks_in_group) == _world.default_pg.size()):
                split_from = _world.default_pg._get_backend(_get_pg_default_device())
                while isinstance(split_from, _ProcessGroupWrapper):
                    split_from = split_from.wrapped_pg
                if split_from:
                    pg_options.split_from = split_from
                    pg_options.split_color = _process_group_color(global_ranks_in_group)
            backend_class = ProcessGroupNCCL(backend_prefix_store, group_rank, group_size, pg_options)
            backend_type = ProcessGroup.BackendType.NCCL
        elif backend_str == Backend.UCC and is_ucc_available():
            backend_class = ProcessGroupUCC(backend_prefix_store, group_rank, group_size, timeout=timeout)
            backend_type = ProcessGroup.BackendType.UCC
        else:
            assert backend_str.upper() in Backend._plugins, f'Unknown c10d backend type {backend_str.upper()}'
            backend_plugin = Backend._plugins[backend_str.upper()]
            creator_fn = backend_plugin.creator_fn
            extended_api = backend_plugin.extended_api
            backend_type = ProcessGroup.BackendType.CUSTOM
            if not extended_api:
                backend_class = creator_fn(backend_prefix_store, group_rank, group_size, timeout)
            else:
                dist_backend_opts = _DistributedBackendOptions()
                dist_backend_opts.store = backend_prefix_store
                dist_backend_opts.group_rank = group_rank
                dist_backend_opts.group_size = group_size
                dist_backend_opts.timeout = timeout
                dist_backend_opts.group_id = group_name
                dist_backend_opts.global_ranks_in_group = global_ranks_in_group
                backend_class = creator_fn(dist_backend_opts, pg_options)
        if backend_str in [Backend.GLOO, Backend.NCCL]:
            backend_class._set_sequence_number_for_group()
        if issubclass(type(backend_class), ProcessGroup):
            pg = backend_class
            break
        if backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]:
            if get_debug_level() == DebugLevel.DETAIL:
                if not _GLOO_AVAILABLE:
                    logger.info('TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but\n                                GLOO is not available. Build with Gloo to\n                                create a wrapper process group in debug mode\n                                to aid collective desynchronization debugging.')
                else:
                    backend_class = _create_process_group_wrapper(wrapped_pg=backend_class, store_prefix=group_name, store=backend_prefix_store, rank=group_rank, world_size=group_size, timeout=timeout)
        if len(set(backend_config.get_device_backend_map().values())) == 1:
            for device in backend_config.get_device_backend_map().keys():
                pg._register_backend(torch.device(device), backend_type, backend_class)
            break
        pg._register_backend(torch.device(device), backend_type, backend_class)
    assert group_name is not None
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    pg._set_group_name(group_name)
    _world.pg_backend_config[pg] = str(backend_config)
    if pg_tag in [None, '']:
        pg_tag = f'ptd:{group_name}'
        _world.tags_to_pg.setdefault('', []).append(pg)
    else:
        pg_tag = f'user:{pg_tag}'
    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return (pg, prefix_store)