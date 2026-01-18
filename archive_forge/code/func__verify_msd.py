import copy
from itertools import chain
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
def _verify_msd(self, msd: Dict[str, Any], dist_msd: Dict[str, Any], options: StateDictOptions=StateDictOptions()) -> None:
    if not options.ignore_frozen_params:
        self.assertEqual(len(msd), len(dist_msd))
    for fqn, param in msd.items():
        dist_param = dist_msd.get(fqn, None)
        if not options.ignore_frozen_params:
            self.assertIsNotNone(dist_param)
            self._compare_tensor(param, dist_param)
        elif dist_param is None:
            self.assertFalse(param.requires_grad)