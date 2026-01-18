from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, cast
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import SWALR
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import LRScheduler
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies.fsdp import FSDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
@staticmethod
def _clear_schedulers(trainer: 'pl.Trainer') -> None:
    if trainer.lr_scheduler_configs:
        assert len(trainer.lr_scheduler_configs) == 1
        trainer.lr_scheduler_configs.clear()