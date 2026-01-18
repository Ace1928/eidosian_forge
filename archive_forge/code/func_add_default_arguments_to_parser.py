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
def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
    """Adds default arguments to the parser."""
    parser.add_argument('--seed_everything', type=Union[bool, int], default=self.seed_everything_default, help='Set to an int to run seed_everything with this value before classes instantiation.Set to True to use a random seed.')