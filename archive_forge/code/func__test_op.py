import itertools
import sys
from functools import wraps
from typing import (
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
from torch.distributed._tensor import (
from torch.distributed._tensor.placement_types import Placement
def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
    out = op_call(*args, **kwargs)
    dtc = DTensorConverter(mesh, args, kwargs)
    for d_args, d_kwargs in dtc:
        self.assertEqual(dtc.successful(), True)
        d_out = op_call(*d_args, **d_kwargs)
        self.assertEqual(d_out.full_tensor(), out)