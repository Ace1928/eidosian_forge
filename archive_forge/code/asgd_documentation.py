import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
from torch._utils import is_compiling
from typing import List, Optional
Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        