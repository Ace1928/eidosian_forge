import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
@staticmethod
def _sync_param_groups(src_param_groups: List[Dict[Any, Any]], dst_param_groups: List[Dict[Any, Any]]) -> None:
    """
        Sync the attributes from the source parameter groups to the destination parameter groups.

        Example attributes include learning rate or scheduler attributes. The
        two parameter groups should have the same length (i.e. same number of
        parameter groups).

        Arguments:
            src_param_groups (list[dict]): parameter groups giving the
                attribute settings to copy.
            dst_param_groups (list[dict]): parameter groups giving the
                attribute settings to set.
        """
    assert len(src_param_groups) == len(dst_param_groups), 'Mismatch between number of source and destination parameter groups'
    for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
        for attr in filter(lambda x: x != 'params', src_param_group.keys()):
            dst_param_group[attr] = src_param_group[attr]