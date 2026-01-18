import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
    """
        If found_inf is 1.0 (True), then scale is multiplied by backoff_factor and growth_tracker is set to zero.
        Otherwise, scale is multiplied by the growth factor when the growth interval is reached.
        """
    assert self._scale is not None and self._growth_tracker is not None
    if found_inf.item() >= 1.0:
        self._scale *= self._backoff_factor
        self._growth_tracker.fill_(0)
    else:
        successful = self._growth_tracker + 1
        if successful == self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker.fill_(0)
        else:
            self._growth_tracker = successful