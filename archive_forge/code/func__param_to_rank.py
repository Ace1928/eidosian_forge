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
@property
def _param_to_rank(self) -> Dict[torch.Tensor, int]:
    """:class:`dict` mapping parameters to their assigned data parallel rank in the partition."""
    if len(self._param_to_rank_cache) == 0:
        for rank, param_groups in enumerate(self._partition_parameters()):
            for param_group in param_groups:
                for param in param_group['params']:
                    self._param_to_rank_cache[param] = rank
    return self._param_to_rank_cache