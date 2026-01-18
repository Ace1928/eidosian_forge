import functools
import warnings
from typing import Callable, Optional, Tuple, Union
import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
def _validate_tp_mesh_dim(device_mesh: DeviceMesh) -> None:
    """
    Check whether TP mesh dimension is valid or not.

    Args:
        device_mesh (:class:`DeviceMesh`):
            The `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        `True` if the mesh dimension
        is valid, `False` otherwise.
    """
    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh:
        if parent_mesh.ndim != 2:
            raise RuntimeError(f'Found TP device_mesh has a parent mesh with dims {parent_mesh.ndim}', 'Currently we only support 2D TP composition with DP.')
        tp_mesh_dim = _mesh_resources.get_parent_mesh_dim(device_mesh)
        if tp_mesh_dim != 1:
            raise RuntimeError(f'Found TP device_mesh on the {tp_mesh_dim} dimension of its parent mesh.', 'Currently we only support intranode TP and TP needs to be the innermost dimension on its parent mesh.')