import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Set, Tuple, Type
import torch
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
def _check_no_key(key: str) -> bool:
    if trainer.lr_scheduler_configs:
        return any((key not in config.scheduler.optimizer.defaults for config in trainer.lr_scheduler_configs))
    return any((key not in optimizer.defaults for optimizer in trainer.optimizers))