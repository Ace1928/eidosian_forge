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
def _share_and_gather(self, store, group_rank: int, group_world_size: int, spec: WorkerSpec) -> List:
    agent_role_info = _RoleInstanceInfo(spec.role, group_rank, spec.local_world_size)
    key_prefix = 'torchelastic/role_info'
    agent_config_enc = agent_role_info.serialize()
    role_infos_bytes = store_util.synchronize(store, agent_config_enc, group_rank, group_world_size, key_prefix)
    role_infos = [_RoleInstanceInfo.deserialize(role_info_bytes) for role_info_bytes in role_infos_bytes]
    return role_infos