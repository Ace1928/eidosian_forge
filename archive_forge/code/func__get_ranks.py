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
def _get_ranks(self, role_infos: List[_RoleInstanceInfo], role_idx: int, start_idx: int=0, end_idx: int=-1) -> Tuple[int, List[int]]:
    if end_idx == -1:
        end_idx = len(role_infos)
    prefix_sum = 0
    total_sum = 0
    for idx in range(start_idx, end_idx):
        if role_idx > idx:
            prefix_sum += role_infos[idx].local_world_size
        total_sum += role_infos[idx].local_world_size
    return (total_sum, list(range(prefix_sum, prefix_sum + role_infos[role_idx].local_world_size)))