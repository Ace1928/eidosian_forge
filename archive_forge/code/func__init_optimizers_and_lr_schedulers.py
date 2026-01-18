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
def _init_optimizers_and_lr_schedulers(model: 'pl.LightningModule') -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
    """Calls `LightningModule.configure_optimizers` and parses and validates the output."""
    from pytorch_lightning.trainer import call
    optim_conf = call._call_lightning_module_hook(model.trainer, 'configure_optimizers', pl_module=model)
    if optim_conf is None:
        rank_zero_warn('`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer')
        optim_conf = _MockOptimizer()
    optimizers, lr_schedulers, monitor = _configure_optimizers(optim_conf)
    lr_scheduler_configs = _configure_schedulers_automatic_opt(lr_schedulers, monitor) if model.automatic_optimization else _configure_schedulers_manual_opt(lr_schedulers)
    _validate_multiple_optimizers_support(optimizers, model)
    _validate_optimizers_attached(optimizers, lr_scheduler_configs)
    _validate_scheduler_api(lr_scheduler_configs, model)
    return (optimizers, lr_scheduler_configs)