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
def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
    """Forward computation using single call to `update`.

        This can be done when the global metric state is a sinple reduction of batch states. This can be unsafe for
        certain metric cases but is also the fastest way to both accumulate globally and compute locally.

        """
    global_state = {attr: getattr(self, attr) for attr in self._defaults}
    _update_count = self._update_count
    self.reset()
    self._to_sync = self.dist_sync_on_step
    self._should_unsync = False
    _temp_compute_on_cpu = self.compute_on_cpu
    self.compute_on_cpu = False
    self._enable_grad = True
    self.update(*args, **kwargs)
    batch_val = self.compute()
    self._update_count = _update_count + 1
    with torch.no_grad():
        self._reduce_states(global_state)
    self._is_synced = False
    self._should_unsync = True
    self._to_sync = self.sync_on_compute
    self._computed = None
    self._enable_grad = False
    self.compute_on_cpu = _temp_compute_on_cpu
    if self.compute_on_cpu:
        self._move_list_states_to_cpu()
    return batch_val