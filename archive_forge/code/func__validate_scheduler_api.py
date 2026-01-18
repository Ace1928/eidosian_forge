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
def _validate_scheduler_api(lr_scheduler_configs: List[LRSchedulerConfig], model: 'pl.LightningModule') -> None:
    for config in lr_scheduler_configs:
        scheduler = config.scheduler
        if not isinstance(scheduler, _Stateful):
            raise TypeError(f'The provided lr scheduler `{scheduler.__class__.__name__}` is invalid. It should have `state_dict` and `load_state_dict` methods defined.')
        if not isinstance(scheduler, LRSchedulerTypeTuple) and (not is_overridden('lr_scheduler_step', model)) and model.automatic_optimization:
            raise MisconfigurationException(f"The provided lr scheduler `{scheduler.__class__.__name__}` doesn't follow PyTorch's LRScheduler API. You should override the `LightningModule.lr_scheduler_step` hook with your own logic if you are using a custom LR scheduler.")