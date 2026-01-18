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
def _check_duplicates_and_update_name(self, optimizer: Optimizer, name: str, seen_optimizers: List[Optimizer], seen_optimizer_types: DefaultDict[Type[Optimizer], int], lr_scheduler_config: Optional[LRSchedulerConfig]) -> List[str]:
    seen_optimizers.append(optimizer)
    optimizer_cls = type(optimizer)
    if lr_scheduler_config is None or lr_scheduler_config.name is None:
        seen_optimizer_types[optimizer_cls] += 1
    param_groups = optimizer.param_groups
    duplicates = self._duplicate_param_group_names(param_groups)
    if duplicates:
        raise MisconfigurationException(f'A single `Optimizer` cannot have multiple parameter groups with identical `name` values. {name} has duplicated parameter group names {duplicates}')
    name = self._add_prefix(name, optimizer_cls, seen_optimizer_types)
    return [self._add_suffix(name, param_groups, i) for i in range(len(param_groups))]