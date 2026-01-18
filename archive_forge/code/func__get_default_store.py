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
def _get_default_store():
    """Get the default store created by init_process_group."""
    if not is_initialized():
        raise ValueError('Default process group has not been initialized, please make sure to call init_process_group.')
    default_pg = _get_default_group()
    _, default_store = _world.pg_map[default_pg]
    return default_store