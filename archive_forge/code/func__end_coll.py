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
@classmethod
def _end_coll(cls, collective, pg):
    with cls._coll_lock:
        if pg.pg_name in cls._cur_coll_on_pgs and cls._cur_coll_on_pgs[pg.pg_name] == collective:
            cls._cur_coll_on_pgs.pop(pg.pg_name)