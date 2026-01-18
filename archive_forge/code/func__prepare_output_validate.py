import functools
import warnings
from typing import Callable, Optional, Tuple, Union
import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
def _prepare_output_validate(_prepare_output_func: _PrepareOutputType) -> _PrepareOutputType:
    """
    Inject common validation logics for _prepare_output funcs via this decorator.

    Include verifying that output needs to be a DTensor
    and only 1D Device Mesh is passed in.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> @_prepare_output_validate
        >>> def make_output_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> # xdoctest: +SKIP(failing)
        >>> dt = distribute(tensor, device_mesh, [Shard(0)])
        >>> make_output_shard_1d(dt, device_mesh, 1)
        >>> # This will call '_prepare_output_validate' first

    Args:
        _prepare_output_func (Callable): The func we want to inject the
            validation into.
    Return:
        func (Callable): Same input func with validation logic added.
    """

    @functools.wraps(_prepare_output_func)
    def wrapper(*args, **kwargs):
        assert len(args) >= 1, '_prepare_output needs at least one arg.'
        output = args[0]
        assert isinstance(output, DTensor), f'Expect output of Tensor Parallel to be a DTensor, but found {type(output)}.'
        if len(args) < 2 or args[1] is None:
            device_mesh = output.device_mesh
            args = (*args[:1], device_mesh, *args[2:])
        else:
            device_mesh = args[1]
        assert device_mesh.ndim == 1, f'device_mesh has dims {device_mesh.ndim} but expected to be 1 for output.'
        return _prepare_output_func(*args, **kwargs)
    return wrapper