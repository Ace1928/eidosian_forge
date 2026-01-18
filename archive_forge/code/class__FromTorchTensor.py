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
class _FromTorchTensor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, device_mesh: DeviceMesh, placements: Tuple[Placement, ...], run_check: bool, shape: Optional[torch.Size]=None, stride: Optional[Tuple[int, ...]]=None) -> 'DTensor':
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh
        if shape and stride:
            tensor_shape, tensor_stride = (shape, stride)
        elif not shape and (not stride):
            global_shape, global_stride = compute_global_tensor_info(input, device_mesh, placements)
            tensor_shape, tensor_stride = (torch.Size(global_shape), tuple(global_stride))
        else:
            raise RuntimeError(f'Found shape:{shape}, stride:{stride}.', 'Please pass both shape and stride at the same time.')
        if device_mesh.get_coordinate() is None:
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    input = input.contiguous()
                    mesh_broadcast(input, device_mesh, mesh_dim=idx)
        dist_tensor = DTensor(input.view_as(input), device_mesh, placements, shape=tensor_shape, dtype=input.dtype, requires_grad=input.requires_grad, stride=tensor_stride)
        return dist_tensor

    @staticmethod
    def backward(ctx, grad_output: 'DTensor'):
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        if grad_output.placements != previous_placement:
            grad_output = Redistribute.apply(grad_output, previous_device_mesh, previous_placement)
        return (grad_output.to_local(), None, None, None, None, None)