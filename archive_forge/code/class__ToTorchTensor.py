import warnings
from typing import Callable, cast, Optional, Sequence, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.dispatch as op_dispatch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor._utils import compute_global_tensor_info
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.random import (
from torch.distributed._tensor.redistribute import (
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
class _ToTorchTensor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: 'DTensor', grad_placements: Optional[Sequence[Placement]], async_output: bool):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor
        if not async_output and isinstance(local_tensor, funcol.AsyncCollectiveTensor):
            local_tensor = local_tensor.wait()
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta
        if grad_placements is not None:
            grad_spec = DTensorSpec(mesh, grad_placements)
            grad_output = redistribute_local_tensor(grad_output, grad_spec, dtensor_spec)
        _, tensor_stride = compute_global_tensor_info(grad_output, mesh, dtensor_spec.placements)
        return (DTensor(grad_output, mesh, dtensor_spec.placements, shape=dtensor_meta.shape, dtype=dtensor_meta.dtype, requires_grad=grad_output.requires_grad, stride=tuple(tensor_stride)), None, None)