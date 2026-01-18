import functools
import torch
import torch.distributed as dist
from typing import Optional
class LowPrecisionState(DefaultState):
    """
    Stores state needed to perform gradient communication in a lower precision within a communication hook.

    Communication hook will cast gradients back to the original
    parameter precision specified by ``parameter_type`` (default: torch.float32).
    Builds on top of the :class:`DefaultState`.

    Args:
        parameter_type (torch.dtype): The precision of model's parameters.
        Required for a hook to cast gradients back to a parameter's precision.
    """
    __slots__ = ['parameter_type']

    def __init__(self, process_group, parameter_type=torch.float32):
        super().__init__(process_group)
        self.parameter_type = parameter_type