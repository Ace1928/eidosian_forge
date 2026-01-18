import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
class ThreadLocalWorld:
    _world = threading.local()

    def _get_world(self) -> WorldData:
        if not hasattr(ThreadLocalWorld._world, 'world'):
            ThreadLocalWorld._world.world = WorldData(None, {}, {}, {}, {}, 0, {}, {}, {}, {})
        return ThreadLocalWorld._world.world

    @property
    def default_pg(self):
        return self._get_world().default_pg

    @default_pg.setter
    def default_pg(self, value):
        self._get_world().default_pg = value

    @property
    def pg_map(self):
        return self._get_world().pg_map

    @property
    def pg_names(self):
        return self._get_world().pg_names

    @property
    def pg_group_ranks(self):
        return self._get_world().pg_group_ranks

    @property
    def pg_backend_config(self):
        return self._get_world().pg_backend_config

    @property
    def group_count(self) -> int:
        return self._get_world().group_count

    @group_count.setter
    def group_count(self, value):
        self._get_world().group_count = value

    @property
    def tags_to_pg(self):
        return self._get_world().tags_to_pg

    @property
    def pg_to_tag(self):
        return self._get_world().pg_to_tag

    @property
    def pg_coalesce_state(self) -> Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]:
        return self._get_world().pg_coalesce_state

    @property
    def pg_default_device(self) -> Dict[dist.ProcessGroup, torch.device]:
        return self._get_world().pg_default_device