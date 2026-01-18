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
def _partition_parameters(self, params_per_rank: Optional[List[List[torch.Tensor]]]=None) -> List[List[Dict]]:
    """
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank (list[list[torch.Tensor]], optional): a
                :class:`list` of length world size containing :class:`list` s
                of parameters to assign to each rank; this provides a way to
                specify a partition manually.
                If ``None``, the parameters are partitioned according to an
                internal algorithm.
                (default: ``None``)

        Returns:
            A :class:`list` where each element of the list contains the
            ``param_groups`` for a rank (which itself is a :class:`list` of
            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank
            stores the ``param_groups`` for all ranks for the collective
            communication in :meth:`step`.

        Raises:
            ValueError: see :meth:`_validate_params_per_rank`.
            RuntimeError: if ``params_per_rank`` is not ``None`` and this
                :class:`ZeroRedundancyOptimizer` instance is using more than
                one parameter group.
        """
    if params_per_rank is None:
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [[] for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_group_params_per_rank: List[List] = [[] for _ in range(self.world_size)]
                params_sorted = sorted(param_group['params'], key=lambda t: t.numel(), reverse=True)
                for param in params_sorted:
                    rank = self._get_min_index(sizes)
                    param_group_params_per_rank[rank].append(param)
                    sizes[rank] += param.numel()
                self._partition_param_group(param_group, param_group_params_per_rank)
        return self._partition_parameters_cache
    assert len(self._partition_parameters_cache) == 0, 'Specifying `params_per_rank` should only be done when the parameters have not been partitioned yet'
    if len(self.param_groups) != 1:
        raise RuntimeError('Specifying `params_per_rank` only supports a single parameter group')
    self._verify_params_per_rank(params_per_rank)
    self._partition_parameters_cache = [[] for _ in range(self.world_size)]
    param_group = self.param_groups[0]
    self._partition_param_group(param_group, params_per_rank)
    return self._partition_parameters_cache