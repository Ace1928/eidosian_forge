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
def _device_to_params_per_rank(self) -> Dict[torch.device, List[List[torch.Tensor]]]:
    """
        Return device parameters assigned per rank.

        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
    assert self.parameters_as_bucket_view, '`_device_to_params_per_rank` should only be used if `parameters_as_bucket_view=True`'
    if len(self._device_to_params_per_rank_cache) == 0:
        for rank, param_groups in enumerate(self._partition_parameters()):
            for param_group in param_groups:
                for param in param_group['params']:
                    device = param.device
                    if device not in self._device_to_params_per_rank_cache:
                        self._device_to_params_per_rank_cache[device] = [[] for _ in range(self.world_size)]
                    self._device_to_params_per_rank_cache[device][rank].append(param)
    return self._device_to_params_per_rank_cache