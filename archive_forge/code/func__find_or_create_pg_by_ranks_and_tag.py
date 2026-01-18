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
def _find_or_create_pg_by_ranks_and_tag(tag: str, ranks: List[int], stride: int) -> ProcessGroup:
    assert len(ranks) % stride == 0, f'Ranks length ({len(ranks)}) must be divisible by stride ({stride})'
    my_rank = get_rank()
    my_ranks = None
    if stride == len(ranks):
        my_ranks = ranks.copy()
        assert my_rank in my_ranks, "rankset doesn't include the current node"
    else:
        for i in range(0, len(ranks), stride):
            rank_set = ranks[i:i + stride]
            if my_rank in rank_set:
                my_ranks = rank_set
        assert my_ranks is not None, "rankset doesn't include the current node"
    my_ranks.sort()
    pg = _find_pg_by_ranks_and_tag(tag, my_ranks)
    if pg is not None:
        return pg
    if tag == '':
        raise ValueError('Cannot automatically create PG with empty tag')
    return _new_group_with_tag(my_ranks, pg_tag=tag)