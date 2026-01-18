from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import do_nothing_closure
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.optimization.closure import OutputResult
from pytorch_lightning.loops.progress import _Progress, _ReadyCompletedTracker
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
class _ManualOptimization(_Loop):
    """A special loop implementing what is known in Lightning as Manual Optimization where the optimization happens
    entirely in the :meth:`~pytorch_lightning.core.LightningModule.training_step` and therefore the user is responsible
    for back-propagating gradients and making calls to the optimizers.

    This loop is a trivial case because it performs only a single iteration (calling directly into the module's
    :meth:`~pytorch_lightning.core.LightningModule.training_step`) and passing through the output(s).

    """
    output_result_cls = ManualResult

    def __init__(self, trainer: 'pl.Trainer') -> None:
        super().__init__(trainer)
        self.optim_step_progress = _Progress.from_defaults(_ReadyCompletedTracker)
        self._output: _OUTPUTS_TYPE = {}

    def run(self, kwargs: OrderedDict) -> _OUTPUTS_TYPE:
        self.on_run_start()
        with suppress(StopIteration):
            self.advance(kwargs)
        self._restarting = False
        return self.on_run_end()

    def on_run_start(self) -> None:
        for lightning_optimizer in self.trainer.strategy._lightning_optimizers:
            lightning_optimizer._on_before_step = self._on_before_step
            lightning_optimizer._on_after_step = self._on_after_step

    def advance(self, kwargs: OrderedDict) -> None:
        """Performs the training step for manual optimization.

        Args:
            kwargs: The kwargs passed down to the hooks.

        """
        trainer = self.trainer
        training_step_output = call._call_strategy_hook(trainer, 'training_step', *kwargs.values())
        del kwargs
        self.trainer.strategy.post_training_step()
        result = self.output_result_cls.from_training_step_output(training_step_output)
        self._output = result.asdict()

    def on_run_end(self) -> _OUTPUTS_TYPE:
        """Returns the result of this loop, i.e., the post-processed outputs from the training step."""
        output, self._output = (self._output, {})
        for lightning_optimizer in self.trainer.strategy._lightning_optimizers:
            lightning_optimizer._on_before_step = do_nothing_closure
            lightning_optimizer._on_after_step = do_nothing_closure
        return output

    def _on_before_step(self) -> None:
        self.optim_step_progress.increment_ready()
        self.trainer.profiler.start('optimizer_step')

    def _on_after_step(self) -> None:
        self.trainer.profiler.stop('optimizer_step')
        self.optim_step_progress.increment_completed()