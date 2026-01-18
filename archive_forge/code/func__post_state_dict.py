import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Union, overload
import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
def _post_state_dict(self, state_dict) -> Dict[str, Any]:
    if isinstance(self.module, FSDP):
        FSDP.optim_state_dict(self.module, self._optimizer, state_dict)
    return state_dict