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
def _prepare_subcommand_parser(self, klass: Type, subcommand: str, **kwargs: Any) -> LightningArgumentParser:
    parser = self.init_parser(**kwargs)
    self._add_arguments(parser)
    skip: Set[Union[str, int]] = set(self.subcommands()[subcommand])
    added = parser.add_method_arguments(klass, subcommand, skip=skip)
    self._subcommand_method_arguments[subcommand] = added
    return parser