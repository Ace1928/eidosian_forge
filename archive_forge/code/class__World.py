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
class _World:
    """
    Container class for c10d process group state.

    This is used during registration and lookup of PG state.

    .. warning:: This is an experimental API intended to expose the inner workings
       of c10d and is subject to change..
    """

    def __init__(self):
        self._default_pg = None
        self._pg_coalesce_state: Dict[ProcessGroup, List[Union[_CollOp, P2POp]]] = {}
        self._pg_default_device: Dict[ProcessGroup, torch.device] = {}

    @property
    def default_pg(self):
        """
        Process group that includes all ranks of the cluster.

        This default ProcessGroup is used by c10d APIs when a ProcessGroup is needed
        but None is provided.
        """
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value):
        self._default_pg = value

    @property
    def pg_map(self) -> Dict[ProcessGroup, Tuple[str, Optional[Store]]]:
        """
        Provide Mapping from ProcessGroup to backend name and store.

        For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
        For MPI pg, it is a map from ProcessGroup to (Backend, None)

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_map
        return _pg_map

    @property
    def pg_names(self) -> Dict[ProcessGroup, str]:
        """
        Process group's names, map from ProcessGroup to str.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_names
        return _pg_names

    @property
    def pg_group_ranks(self) -> Dict[ProcessGroup, Dict[int, int]]:
        """
        Process group's global rank to local rank mapping.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_backend_config(self) -> Dict[ProcessGroup, str]:
        """
        Process group's backend config.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_backend_config
        return _pg_backend_config

    @property
    def group_count(self) -> int:
        """
        Process group count for default naming.

        TODO don't expose group_count, use something else instead
        """
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value):
        """Use to compute the name of ProcessGroups when using global synchronization."""
        global _group_count
        _group_count = value

    @property
    def tags_to_pg(self) -> Dict[str, List[ProcessGroup]]:
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> Dict[ProcessGroup, str]:
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_coalesce_state(self) -> Dict[ProcessGroup, List[Union[_CollOp, P2POp]]]:
        return self._pg_coalesce_state

    @property
    def pg_default_device(self) -> Dict[ProcessGroup, torch.device]:
        return self._pg_default_device

    @property
    def pg_config_info(self) -> List[Dict[str, Union[int, str]]]:
        """
        Return a list of dict with process groups and backends.

        Along with their unique IDs and configurations (types and ranks).
        """
        config_info = []
        default_pg_size = _get_group_size(None)
        for pg, backend in self.pg_map.items():
            backend_type = Backend.backend_type_map[backend[0]]
            ranks = self.pg_group_ranks[pg]
            config_info.append({'pg_name': self.pg_names[pg], 'backend_id': pg._backend_id(backend_type), 'backend_config': self.pg_backend_config[pg], 'ranks': list(ranks.keys()) if len(ranks) != default_pg_size else [], 'group_size': len(ranks), 'group_count': self.group_count})
        return config_info