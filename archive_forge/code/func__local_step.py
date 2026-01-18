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
def _local_step(self, gradients: Optional[List[Optional[torch.Tensor]]]=None, closure: Optional[Callable[[], float]]=None, **kwargs: Any) -> Optional[float]:
    """
        Perform a single optimizer step without syncing parameters across ranks.

        Arguments:
            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`
                of length equal to the number of parameters assigned to this
                rank containing gradient tensors or ``None`` as its elements;
                a ``None`` in the :class:`list` indicates that the
                corresponding parameter should not be updated.
                If the argument itself is ``None``, then all parameters are
                updated, and the gradients are assumed to be already populated.
                (default: ``None``)
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers and should be
                ``None`` if ``gradients`` is not ``None``; (default: ``None``)
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. warning::
            The argument ``gradients`` should only be specified (i.e. not
            ``None``) if ``overlap_with_ddp=True``, in which case
            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.
        """
    Join.notify_join_context(self)
    is_trainable_mask = self._get_is_trainable_mask()
    if is_trainable_mask != self._is_trainable_mask:
        if self._overlap_with_ddp:
            raise RuntimeError('ZeroRedundancyOptimizer with `overlap_with_ddp=True` does not support changing parameter trainability at run time')
        logger.warning('ZeroRedundancyOptimizer detected that the trainable parameters changed; rebuilding the parameter buckets if enabled')
        self._build_param_buckets()
        self._is_trainable_mask = is_trainable_mask
    self._sync_param_groups(self.param_groups, self.optim.param_groups)
    if gradients is None:
        loss = self.optim.step(**kwargs) if closure is None else self.optim.step(closure=closure, **kwargs)
    else:
        assert self._overlap_with_ddp, 'Specifying `gradients` should not be used when `overlap_with_ddp=False`'
        assert closure is None, '`closure` is not supported when using a local functional optimizer'
        loss = self.optim.step(gradients=gradients)
    self._sync_param_groups(self.optim.param_groups, self.param_groups)
    return loss