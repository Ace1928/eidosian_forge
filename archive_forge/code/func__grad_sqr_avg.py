import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import numpy as np
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.optim import SGD, Optimizer
def _grad_sqr_avg(self, pg_idx: Optional[int]=None) -> float:
    """
        Current estimate of the squared l2-norm of the true gradient
        (sigma squared in the AdaScale paper).

        Args:
            pg_idx (Optional[int]):
                Optional index for a parameter group.

        Returns:
            (float):
                Estimate of squared l2-norm.
        """
    if pg_idx is not None:
        return self._state['grad_sqr_avg'][pg_idx]
    else:
        return float(np.sum(self._state['grad_sqr_avg']))