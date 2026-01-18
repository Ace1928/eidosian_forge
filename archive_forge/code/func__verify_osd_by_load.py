import copy
from itertools import chain
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
def _verify_osd_by_load(self, model: nn.Module, optim: torch.optim.Optimizer, new_optim: torch.optim.Optimizer, dist_osd: Dict[str, Any]) -> None:
    new_dist_osd = _gather_state_dict(dist_osd)
    set_state_dict(model, optimizers=new_optim, model_state_dict={}, optim_state_dict=new_dist_osd)
    self.assertEqual(optim.state_dict(), new_optim.state_dict())