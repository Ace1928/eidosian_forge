import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """
    __slots__ = ['role', 'rank', 'local_world_size']

    def __init__(self, role: str, rank: int, local_world_size: int):
        """Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {'role': self.role, 'rank': self.rank, 'local_world_size': self.local_world_size}
        return json.dumps(dict_data).encode(encoding='UTF-8')

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding='UTF-8'))
        return _RoleInstanceInfo(dict_data['role'], dict_data['rank'], dict_data['local_world_size'])

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        start_idx, end_idx = (-1, -1)
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)