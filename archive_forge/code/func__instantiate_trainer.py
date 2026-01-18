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
def _instantiate_trainer(self, config: Dict[str, Any], callbacks: List[Callback]) -> Trainer:
    key = 'callbacks'
    if key in config:
        if config[key] is None:
            config[key] = []
        elif not isinstance(config[key], list):
            config[key] = [config[key]]
        config[key].extend(callbacks)
        if key in self.trainer_defaults:
            value = self.trainer_defaults[key]
            config[key] += value if isinstance(value, list) else [value]
        if self.save_config_callback and (not config.get('fast_dev_run', False)):
            config_callback = self.save_config_callback(self._parser(self.subcommand), self.config.get(str(self.subcommand), self.config), **self.save_config_kwargs)
            config[key].append(config_callback)
    else:
        rank_zero_warn(f'The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will not be included.')
    return self.trainer_class(**config)