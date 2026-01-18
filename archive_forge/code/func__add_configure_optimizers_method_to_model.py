import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
    """Overrides the model's :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers` method if a
        single optimizer and optionally a scheduler argument groups are added to the parser as 'AUTOMATIC'."""
    if not self.auto_configure_optimizers:
        return
    parser = self._parser(subcommand)

    def get_automatic(class_type: Union[Type, Tuple[Type, ...]], register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]]) -> List[str]:
        automatic = []
        for key, (base_class, link_to) in register.items():
            if not isinstance(base_class, tuple):
                base_class = (base_class,)
            if link_to == 'AUTOMATIC' and any((issubclass(c, class_type) for c in base_class)):
                automatic.append(key)
        return automatic
    optimizers = get_automatic(Optimizer, parser._optimizers)
    lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)
    if len(optimizers) == 0:
        return
    if len(optimizers) > 1 or len(lr_schedulers) > 1:
        raise MisconfigurationException(f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer and one lr_scheduler to be 'AUTOMATIC', but found {optimizers + lr_schedulers}. In this case the user is expected to link the argument groups and implement `configure_optimizers`, see https://lightning.ai/docs/pytorch/stable/common/lightning_cli.html#optimizers-and-learning-rate-schedulers")
    optimizer_class = parser._optimizers[optimizers[0]][0]
    optimizer_init = self._get(self.config_init, optimizers[0])
    if not isinstance(optimizer_class, tuple):
        optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
    if not optimizer_init:
        return
    lr_scheduler_init = None
    if lr_schedulers:
        lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
        lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
        if not isinstance(lr_scheduler_class, tuple):
            lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)
    if is_overridden('configure_optimizers', self.model):
        _warn(f'`{self.model.__class__.__name__}.configure_optimizers` will be overridden by `{self.__class__.__name__}.configure_optimizers`.')
    optimizer = instantiate_class(self.model.parameters(), optimizer_init)
    lr_scheduler = instantiate_class(optimizer, lr_scheduler_init) if lr_scheduler_init else None
    fn = partial(self.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
    update_wrapper(fn, self.configure_optimizers)
    self.model.configure_optimizers = MethodType(fn, self.model)