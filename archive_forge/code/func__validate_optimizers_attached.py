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
def _validate_optimizers_attached(optimizers: List[Optimizer], lr_scheduler_configs: List[LRSchedulerConfig]) -> None:
    for config in lr_scheduler_configs:
        if config.scheduler.optimizer not in optimizers:
            raise MisconfigurationException("Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`.")