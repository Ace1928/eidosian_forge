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
def _process_group_name(ranks, use_hashed_name):
    global _world
    if use_hashed_name:
        pg_name = _hash_ranks(ranks)
        while pg_name in _world.pg_names.values():
            pg_name = hashlib.sha1(bytes(pg_name + '_', 'utf-8')).hexdigest()
    else:
        pg_name = str(_world.group_count)
        _world.group_count += 1
    return pg_name