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
def _check_p2p_op_list(p2p_op_list):
    """
    Check that the ``p2p_op_list`` is a list of P2POp instances.

    Also, check that all ops use the same group.
    """
    if not isinstance(p2p_op_list, list) or not all((isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list)):
        raise ValueError('Invalid ``p2p_op_list``. Each op is expected to to be of type ``torch.distributed.P2POp``.')
    group = p2p_op_list[0].group
    if not all((group == p2p_op.group for p2p_op in p2p_op_list)):
        raise ValueError('All ops need to use the same group.')