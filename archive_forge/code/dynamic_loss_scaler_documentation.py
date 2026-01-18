from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional
import torch

        :meth:`step` unscale the gradients and step the optimizer.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Args:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        Returns:
            The return value of ``optimizer.step(*args, **kwargs)``.  None when overflow or underflow
            gradients occur and optimizer.step() is skipped.
        