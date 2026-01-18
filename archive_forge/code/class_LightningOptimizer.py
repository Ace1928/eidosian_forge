from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
class LightningOptimizer:
    """This class is used to wrap the user optimizers and handle properly the backward and optimizer_step logic across
    accelerators, AMP, accumulate_grad_batches.

    Note: The purpose of this wrapper is only to define new methods and redirect the `.step()` call. The internal
    state ``__dict__`` is not kept in sync with the internal state of the original optimizer, but the Trainer never
    relies on the internal state of the wrapper.

    """

    def __init__(self, optimizer: Optimizer):
        self._optimizer = optimizer
        self._strategy: Optional[pl.strategies.Strategy] = None
        self._on_before_step = do_nothing_closure
        self._on_after_step = do_nothing_closure
        self.__class__ = type('Lightning' + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @contextmanager
    def toggle_model(self, sync_grad: bool=True) -> Generator[None, None, None]:
        """This function is just a helper for advanced users.

        Considering the current optimizer as A and all other optimizers as B.
        Toggling means all parameters from B exclusive to A will have ``requires_grad`` set to False.

        When performing gradient accumulation, there is no need to perform grad synchronization
        during the accumulation phase.
        Setting `sync_grad` to False will block this synchronization and improve performance.

        """
        from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
        assert self._strategy is not None
        lightning_module = self._strategy.lightning_module
        assert lightning_module is not None
        with _block_parallel_sync_behavior(self._strategy, block=not sync_grad):
            lightning_module.toggle_optimizer(self)
            yield
            lightning_module.untoggle_optimizer(self)

    def step(self, closure: Optional[Callable[[], Any]]=None, **kwargs: Any) -> Any:
        """Performs a single optimization step (parameter update).

        Args:
            closure: An optional optimizer closure.
            kwargs: Any additional arguments to the ``optimizer.step()`` call.

        Returns:
            The output from the step call, which is generally the output of the closure execution.

        Example::

            # Scenario for a GAN using manual optimization
            def training_step(self, batch, batch_idx):
                opt_gen, opt_dis = self.optimizers()

                ...

                # compute generator loss
                loss_gen = self.compute_generator_loss(...)
                # zero_grad needs to be called before backward
                opt_gen.zero_grad()
                self.manual_backward(loss_gen)
                opt_gen.step()

                # compute discriminator loss
                loss_dis = self.compute_discriminator_loss(...)

                # zero_grad needs to be called before backward
                opt_dis.zero_grad()
                self.manual_backward(loss_dis)
                opt_dis.step()


            # A more advanced example
            def training_step(self, batch, batch_idx):
                opt_gen, opt_dis = self.optimizers()

                ...
                accumulated_grad_batches = batch_idx % 2 == 0

                # compute generator loss
                def closure_gen():
                    loss_gen = self.compute_generator_loss(...)
                    self.manual_backward(loss_gen)
                    if accumulated_grad_batches:
                        opt_gen.zero_grad()

                with opt_gen.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_gen.step(closure=closure_gen)

                def closure_dis():
                    loss_dis = self.compute_discriminator_loss(...)
                    self.manual_backward(loss_dis)
                    if accumulated_grad_batches:
                        opt_dis.zero_grad()

                with opt_dis.toggle_model(sync_grad=accumulated_grad_batches):
                    opt_dis.step(closure=closure_dis)

        """
        self._on_before_step()
        if closure is None:
            closure = do_nothing_closure
        elif not callable(closure):
            raise MisconfigurationException('When `optimizer.step(closure)` is called, the closure should be callable')
        assert self._strategy is not None
        step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
        self._on_after_step()
        return step_output

    @classmethod
    def _to_lightning_optimizer(cls, optimizer: Union[Optimizer, 'LightningOptimizer'], strategy: 'pl.strategies.Strategy') -> 'LightningOptimizer':
        lightning_optimizer = optimizer if isinstance(optimizer, LightningOptimizer) else cls(optimizer)
        lightning_optimizer._strategy = proxy(strategy)
        return lightning_optimizer

    def __getattr__(self, item: Any) -> Any:
        return getattr(self._optimizer, item)