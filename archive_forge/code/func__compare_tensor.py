import copy
from itertools import chain
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
def _compare_tensor(self, orig_tensor, dist_tensor):
    if isinstance(dist_tensor, (DTensor, ShardedTensor)):
        dist_tensor = _gather_state_dict({'mykey': dist_tensor}).pop('mykey')
    self.assertTrue(isinstance(dist_tensor, torch.Tensor))
    self.assertTrue(torch.allclose(orig_tensor, dist_tensor))