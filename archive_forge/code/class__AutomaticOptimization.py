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
class _AutomaticOptimization(_Loop):
    """Performs automatic optimization (forward, zero grad, backward, optimizer step)"""
    output_result_cls = ClosureResult

    def __init__(self, trainer: 'pl.Trainer') -> None:
        super().__init__(trainer)
        self.optim_progress: _OptimizationProgress = _OptimizationProgress()
        self._skip_backward: bool = False

    def run(self, optimizer: Optimizer, batch_idx: int, kwargs: OrderedDict) -> _OUTPUTS_TYPE:
        """Runs closure (train step + backward) together with optimization if necessary.

        Args:
            kwargs: the kwargs passed down to the hooks
            batch_idx: the current batch index.
            optimizer: the optimizer

        """
        closure = self._make_closure(kwargs, optimizer, batch_idx)
        if not self.trainer.strategy.handles_gradient_accumulation and self.trainer.fit_loop._should_accumulate():
            with _block_parallel_sync_behavior(self.trainer.strategy, block=True):
                closure()
        else:
            self._optimizer_step(batch_idx, closure)
        result = closure.consume_result()
        if result.loss is None:
            return {}
        return result.asdict()

    def _make_closure(self, kwargs: OrderedDict, optimizer: Optimizer, batch_idx: int) -> Closure:
        """Build a closure object that captures the given arguments and runs the `training_step` function and
        optionally other functions such as `backward` and `zero_grad`."""
        step_fn = self._make_step_fn(kwargs)
        backward_fn = self._make_backward_fn(optimizer)
        zero_grad_fn = self._make_zero_grad_fn(batch_idx, optimizer)
        return Closure(step_fn=step_fn, backward_fn=backward_fn, zero_grad_fn=zero_grad_fn)

    def _make_step_fn(self, kwargs: OrderedDict) -> Callable[[], ClosureResult]:
        """Build the step function that runs the `training_step` and processes its output."""
        return partial(self._training_step, kwargs)

    def _make_zero_grad_fn(self, batch_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
        """Build a `zero_grad` function that zeroes the gradients before back-propagation.

        Returns ``None`` in the case backward needs to be skipped.

        """
        if self._skip_backward:
            return None
        is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
        if not is_first_batch_to_accumulate:
            return None

        def zero_grad_fn() -> None:
            self._on_before_zero_grad(optimizer)
            self._optimizer_zero_grad(batch_idx, optimizer)
        return zero_grad_fn

    def _make_backward_fn(self, optimizer: Optimizer) -> Optional[Callable[[Tensor], None]]:
        """Build a `backward` function that handles back-propagation through the output produced by the `training_step`
        function.

        Returns ``None`` in the case backward needs to be skipped.

        """
        if self._skip_backward:
            return None

        def backward_fn(loss: Tensor) -> None:
            call._call_strategy_hook(self.trainer, 'backward', loss, optimizer)
        return backward_fn

    def _optimizer_step(self, batch_idx: int, train_step_and_backward_closure: Callable[[], Optional[Tensor]]) -> None:
        """Performs the optimizer step and some sanity checking.

        Args:
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default, called by the optimizer (if possible)

        """
        trainer = self.trainer
        optimizer = trainer.strategy._lightning_optimizers[0]
        should_accumulate = trainer.fit_loop._should_accumulate()
        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_ready()
        call._call_lightning_module_hook(trainer, 'optimizer_step', trainer.current_epoch, batch_idx, optimizer, train_step_and_backward_closure)
        if not should_accumulate:
            self.optim_progress.optimizer.step.increment_completed()

    def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Calls the ``on_before_zero_grad`` hook.

        Args:
            optimizer: the current optimizer

        """
        trainer = self.trainer
        self.optim_progress.optimizer.zero_grad.increment_ready()
        call._call_callback_hooks(trainer, 'on_before_zero_grad', optimizer)
        call._call_lightning_module_hook(trainer, 'on_before_zero_grad', optimizer)
        self.optim_progress.optimizer.zero_grad.increment_started()

    def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Zeroes out all gradients of parameters optimized by the current optimizer.

        Args:
            batch_idx: the index of the current batch
            optimizer: the current optimizer

        """
        trainer = self.trainer
        call._call_lightning_module_hook(trainer, 'optimizer_zero_grad', trainer.current_epoch, batch_idx, optimizer)
        self.optim_progress.optimizer.zero_grad.increment_completed()

    def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
        """Performs the actual train step with the tied hooks.

        Args:
            kwargs: the kwargs passed down to the hooks.

        Returns:
            A ``ClosureResult`` containing the training step output.

        """
        trainer = self.trainer
        training_step_output = call._call_strategy_hook(trainer, 'training_step', *kwargs.values())
        self.trainer.strategy.post_training_step()
        return self.output_result_cls.from_training_step_output(training_step_output, trainer.accumulate_grad_batches)