import contextlib
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import Precision as FabricPrecision
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities import GradClipAlgorithmType
def _wrap_closure(self, model: 'pl.LightningModule', optimizer: Optimizer, closure: Callable[[], Any]) -> Any:
    """This double-closure allows makes sure the ``closure`` is executed before the ``on_before_optimizer_step``
        hook is called.

        The closure (generally) runs ``backward`` so this allows inspecting gradients in this hook. This structure is
        consistent with the ``Precision`` subclasses that cannot pass ``optimizer.step(closure)`` directly.

        """
    closure_result = closure()
    self._after_closure(model, optimizer)
    return closure_result