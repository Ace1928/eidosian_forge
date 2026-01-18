import functools
import warnings
from typing import Callable, Optional, Tuple, Union
import torch
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
def _prepare_input_validate(_prepare_input_func: _PrepareInputType) -> _PrepareInputType:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either
    a :class:`Tensor` or :class:`DTensor` and only 1D :class:`DeviceMesh`
    is passed in.

    Args:
        _prepare_input_func (Callable): The func we want to inject the
            validation into.

    Returns:
        func (Callable): Same input function with validation logic added.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> @_prepare_input_validate
        >>> def make_input_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> # xdoctest: +SKIP(failing)
        >>> input = torch.rand(...)
        >>> dtensor = make_input_shard_1d(input, device_mesh, 1)
        >>> # This will call '_prepare_input_validate' first
    """

    @functools.wraps(_prepare_input_func)
    def wrapper(*args, **kwargs):
        assert len(args) >= 1, '_prepare_input needs at least one arg.'
        input = args[0]
        if isinstance(input, (list, tuple)):
            input = input[0]
            args = (input, *args[1:])
        device_mesh = None if len(args) < 2 else args[1]
        if device_mesh is None:
            if isinstance(input, DTensor):
                device_mesh = input.device_mesh
                args = (*args[:1], device_mesh, *args[2:])
            else:
                raise RuntimeError('device_mesh is not passed nor can be inferred')
        if device_mesh.ndim != 1:
            raise RuntimeError(f'device_mesh has dims {device_mesh.ndim} but expected to be 1 for input.')
        return _prepare_input_func(*args, **kwargs)
    return wrapper