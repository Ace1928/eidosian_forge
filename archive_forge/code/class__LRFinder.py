import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
class _LRFinder:
    """LR finder object. This object stores the results of lr_find().

    Args:
        mode: either `linear` or `exponential`, how to increase lr after each step

        lr_min: lr to start search from

        lr_max: lr to stop search

        num_training: number of steps to take between lr_min and lr_max

    Example::
        # Run lr finder
        lr_finder = trainer.lr_find(model)

        # Results stored in
        lr_finder.results

        # Plot using
        lr_finder.plot()

        # Get suggestion
        lr = lr_finder.suggestion()

    """

    def __init__(self, mode: str, lr_min: float, lr_max: float, num_training: int) -> None:
        assert mode in ('linear', 'exponential'), 'mode should be either `linear` or `exponential`'
        self.mode = mode
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_training = num_training
        self.results: Dict[str, Any] = {}
        self._total_batch_idx = 0

    def _exchange_scheduler(self, trainer: 'pl.Trainer') -> None:
        """Decorate `trainer.strategy.setup_optimizers` method such that it sets the user's originally specified
        optimizer together with a new scheduler that takes care of the learning rate search."""
        from pytorch_lightning.core.optimizer import _validate_optimizers_attached
        optimizers = trainer.strategy.optimizers
        if len(optimizers) != 1:
            raise MisconfigurationException(f'`model.configure_optimizers()` returned {len(optimizers)}, but learning rate finder only works with single optimizer')
        optimizer = optimizers[0]
        new_lrs = [self.lr_min] * len(optimizer.param_groups)
        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
            param_group['lr'] = new_lr
            param_group['initial_lr'] = new_lr
        args = (optimizer, self.lr_max, self.num_training)
        scheduler = _LinearLR(*args) if self.mode == 'linear' else _ExponentialLR(*args)
        scheduler = cast(LRScheduler, scheduler)
        trainer.strategy.optimizers = [optimizer]
        trainer.strategy.lr_scheduler_configs = [LRSchedulerConfig(scheduler, interval='step')]
        _validate_optimizers_attached(trainer.optimizers, trainer.lr_scheduler_configs)

    def plot(self, suggest: bool=False, show: bool=False, ax: Optional['Axes']=None) -> Optional['plt.Figure']:
        """Plot results from lr_find run
        Args:
            suggest: if True, will mark suggested lr to use with a red point

            show: if True, will show figure

            ax: Axes object to which the plot is to be drawn. If not provided, a new figure is created.
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise MisconfigurationException('To use the `plot` method, you must have Matplotlib installed. Install it by running `pip install -U matplotlib`.')
        import matplotlib.pyplot as plt
        lrs = self.results['lr']
        losses = self.results['loss']
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(lrs, losses)
        if self.mode == 'exponential':
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        if suggest:
            _ = self.suggestion()
            if self._optimal_idx:
                ax.plot(lrs[self._optimal_idx], losses[self._optimal_idx], markersize=10, marker='o', color='red')
        if show:
            plt.show()
        return fig

    def suggestion(self, skip_begin: int=10, skip_end: int=1) -> Optional[float]:
        """This will propose a suggestion for an initial learning rate based on the point with the steepest negative
        gradient.

        Args:
            skip_begin: how many samples to skip in the beginning; helps to avoid too naive estimates
            skip_end: how many samples to skip in the end; helps to avoid too optimistic estimates

        Returns:
            The suggested initial learning rate to use, or `None` if a suggestion is not possible due to too few
            loss samples.

        """
        losses = torch.tensor(self.results['loss'][skip_begin:-skip_end])
        losses = losses[torch.isfinite(losses)]
        if len(losses) < 2:
            log.error('Failed to compute suggestion for learning rate because there are not enough points. Increase the loop iteration limits or the size of your dataset/dataloader.')
            self._optimal_idx = None
            return None
        gradients = torch.gradient(losses)[0]
        min_grad = torch.argmin(gradients).item()
        self._optimal_idx = min_grad + skip_begin
        return self.results['lr'][self._optimal_idx]