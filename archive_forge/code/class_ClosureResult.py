from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, OrderedDict
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.optimization.closure import AbstractClosure, OutputResult
from pytorch_lightning.loops.progress import _OptimizationProgress
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache
from pytorch_lightning.utilities.types import STEP_OUTPUT
@dataclass
class ClosureResult(OutputResult):
    """A container to hold the result of a :class:`Closure` call.

    It is created from the output of :meth:`~pytorch_lightning.core.LightningModule.training_step`.

    Attributes:
        closure_loss: The loss with a graph attached.
        loss: A detached copy of the closure loss.
        extra: Any keys other than the loss returned.

    """
    closure_loss: Optional[Tensor]
    loss: Optional[Tensor] = field(init=False, default=None)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._clone_loss()

    def _clone_loss(self) -> None:
        if self.closure_loss is not None:
            self.loss = self.closure_loss.detach().clone()

    @classmethod
    def from_training_step_output(cls, training_step_output: STEP_OUTPUT, normalize: int=1) -> 'ClosureResult':
        closure_loss, extra = (None, {})
        if isinstance(training_step_output, Mapping):
            closure_loss = training_step_output.get('loss')
            if closure_loss is None:
                raise MisconfigurationException("In automatic_optimization, when `training_step` returns a dict, the 'loss' key needs to be present")
            extra = {k: v for k, v in training_step_output.items() if k != 'loss'}
        elif isinstance(training_step_output, Tensor):
            closure_loss = training_step_output
        elif training_step_output is not None:
            raise MisconfigurationException('In automatic optimization, `training_step` must return a Tensor, a dict, or None (where the step will be skipped).')
        if closure_loss is not None:
            closure_loss = closure_loss / normalize
        return cls(closure_loss, extra=extra)

    @override
    def asdict(self) -> Dict[str, Any]:
        return {'loss': self.loss, **self.extra}