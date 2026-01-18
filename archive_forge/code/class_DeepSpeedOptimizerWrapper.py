import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
class DeepSpeedOptimizerWrapper(AcceleratedOptimizer):
    """
    Internal wrapper around a deepspeed optimizer.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)
        self.__has_overflow__ = hasattr(self.optimizer, 'overflow')

    def zero_grad(self, set_to_none=None):
        pass

    def step(self):
        pass

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        if self.__has_overflow__:
            return self.optimizer.overflow
        return False