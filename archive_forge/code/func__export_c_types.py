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
def _export_c_types():
    _public_types_to_change_module = [AllreduceCoalescedOptions, AllreduceOptions, AllToAllOptions, BarrierOptions, BroadcastOptions, GatherOptions, PrefixStore, ProcessGroup, ReduceOp, ReduceOptions, ReduceScatterOptions, ScatterOptions, Store, DebugLevel, get_debug_level, Work]
    for type in _public_types_to_change_module:
        type.__module__ = 'torch.distributed.distributed_c10d'