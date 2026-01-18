import os
from functools import partial
from typing import Any, Callable
import torch
from typing_extensions import get_args, override
import pytorch_lightning as pl
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.plugins.precision.xla import _PRECISION_INPUT
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision import Precision
from pytorch_lightning.utilities.exceptions import MisconfigurationException
def _xla_wrap_closure(self, optimizer: Optimizable, closure: Callable[[], Any]) -> Any:
    import torch_xla.core.xla_model as xm
    closure_result = closure()
    xm.reduce_gradients(optimizer)
    return closure_result