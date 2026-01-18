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
def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
    """Adds subcommands to the input parser."""
    self._subcommand_parsers: Dict[str, LightningArgumentParser] = {}
    parser_subcommands = parser.add_subcommands()
    trainer_class = self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
    for subcommand in self.subcommands():
        fn = getattr(trainer_class, subcommand)
        description = _get_short_description(fn)
        subparser_kwargs = kwargs.get(subcommand, {})
        subparser_kwargs.setdefault('description', description)
        subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **subparser_kwargs)
        self._subcommand_parsers[subcommand] = subcommand_parser
        parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)