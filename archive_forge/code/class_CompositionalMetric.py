import builtins
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torchmetrics.utilities.data import (
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from torchmetrics.utilities.prints import rank_zero_warn
class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute."""

    def __init__(self, operator: Callable, metric_a: Union[Metric, float, Tensor], metric_b: Union[Metric, float, Tensor, None]) -> None:
        """Class for creating compositions of metrics.

        This metric class is the output of adding, multiplying etc. any other metric. The metric re-implements the
        standard ``update``, ``forward``, ``reset`` and ``compute`` methods to redirect the arguments to the metrics
        that formed this composition.

        Args:
            operator:
                The operator taking in one (if metric_b is None) or two arguments. Will be applied to outputs of
                metric_a.compute() and (optionally if metric_b is not None) metric_b.compute()
            metric_a:
                First metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None.

        """
        super().__init__()
        self.op = operator
        if isinstance(metric_a, Tensor):
            self.register_buffer('metric_a', metric_a, persistent=False)
        else:
            self.metric_a = metric_a
        if isinstance(metric_b, Tensor):
            self.register_buffer('metric_b', metric_b, persistent=False)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Optional[Callable]=None, process_group: Optional[Any]=None) -> None:
        """No syncing required here.

        syncing will be done in metric_a and metric_b.

        """

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Redirect the call to the input which the conposition was formed from."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))
        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:
        """Redirect the call to the input which the conposition was formed from."""
        val_a = self.metric_a.compute() if isinstance(self.metric_a, Metric) else self.metric_a
        val_b = self.metric_b.compute() if isinstance(self.metric_b, Metric) else self.metric_b
        if val_b is None:
            return self.op(val_a)
        return self.op(val_a, val_b)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate metric on current batch and accumulate to global state."""
        val_a = self.metric_a(*args, **self.metric_a._filter_kwargs(**kwargs)) if isinstance(self.metric_a, Metric) else self.metric_a
        val_b = self.metric_b(*args, **self.metric_b._filter_kwargs(**kwargs)) if isinstance(self.metric_b, Metric) else self.metric_b
        if val_a is None:
            self._forward_cache = None
            return self._forward_cache
        if val_b is None:
            if isinstance(self.metric_b, Metric):
                self._forward_cache = None
                return self._forward_cache
            self._forward_cache = self.op(val_a)
            return self._forward_cache
        self._forward_cache = self.op(val_a, val_b)
        return self._forward_cache

    def reset(self) -> None:
        """Redirect the call to the input which the conposition was formed from."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()
        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool=False) -> None:
        """Change if metric state is persistent (save as part of state_dict) or not.

        Args:
            mode: bool indicating if all states should be persistent or not

        """
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        """Return a representation of the compositional metric, including the two inputs it was formed from."""
        _op_metrics = f'(\n  {self.op.__name__}(\n    {self.metric_a!r},\n    {self.metric_b!r}\n  )\n)'
        return self.__class__.__name__ + _op_metrics

    def _wrap_compute(self, compute: Callable) -> Callable:
        """No wrapping necessary for compositional metrics."""
        return compute