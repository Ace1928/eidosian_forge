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
def _configure_schedulers_automatic_opt(schedulers: list, monitor: Optional[str]) -> List[LRSchedulerConfig]:
    """Convert each scheduler into `LRSchedulerConfig` with relevant information, when using automatic optimization."""
    lr_scheduler_configs = []
    for scheduler in schedulers:
        if isinstance(scheduler, dict):
            supported_keys = {field.name for field in fields(LRSchedulerConfig)}
            extra_keys = scheduler.keys() - supported_keys
            if extra_keys:
                rank_zero_warn(f'Found unsupported keys in the lr scheduler dict: {extra_keys}. HINT: remove them from the output of `configure_optimizers`.', category=RuntimeWarning)
                scheduler = {k: v for k, v in scheduler.items() if k in supported_keys}
            if 'scheduler' not in scheduler:
                raise MisconfigurationException('The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler')
            if 'interval' in scheduler and scheduler['interval'] not in ('step', 'epoch'):
                raise MisconfigurationException(f'The "interval" key in lr scheduler dict must be "step" or "epoch" but is "{scheduler['interval']}"')
            scheduler['reduce_on_plateau'] = scheduler.get('reduce_on_plateau', isinstance(scheduler['scheduler'], optim.lr_scheduler.ReduceLROnPlateau))
            if scheduler['reduce_on_plateau'] and scheduler.get('monitor', None) is None:
                raise MisconfigurationException('The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used. For example: {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "your_loss"}}')
            is_one_cycle = isinstance(scheduler['scheduler'], optim.lr_scheduler.OneCycleLR)
            if is_one_cycle and scheduler.get('interval', 'epoch') == 'epoch':
                rank_zero_warn("A `OneCycleLR` scheduler is using 'interval': 'epoch'. Are you sure you didn't mean 'interval': 'step'?", category=RuntimeWarning)
            config = LRSchedulerConfig(**scheduler)
        elif isinstance(scheduler, ReduceLROnPlateau):
            if monitor is None:
                raise MisconfigurationException('`configure_optimizers` must include a monitor when a `ReduceLROnPlateau` scheduler is used. For example: {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}')
            config = LRSchedulerConfig(scheduler, reduce_on_plateau=True, monitor=monitor)
        else:
            config = LRSchedulerConfig(scheduler)
        lr_scheduler_configs.append(config)
    return lr_scheduler_configs