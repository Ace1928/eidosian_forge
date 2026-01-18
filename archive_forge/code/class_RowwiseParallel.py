from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear only.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> ...
        >>> # By default, the input of the "w2" Linear will be annotated to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={"w2": RowwiseParallel()},
        >>> )
        >>> ...
    """

    def __init__(self, *, input_layouts: Optional[Placement]=None, output_layouts: Optional[Placement]=None, use_local_output: bool=True):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.desired_input_layouts = (Shard(-1),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts)
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        if isinstance(module, nn.Linear):
            module.register_parameter('weight', nn.Parameter(distribute_tensor(module.weight, device_mesh, [Shard(1)])))
            if module.bias is not None:
                module.register_parameter('bias', nn.Parameter(distribute_tensor(module.bias, device_mesh, [Replicate()])))
        else:
            raise NotImplementedError('RowwiseParallel currently only support nn.Linear!')

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, outputs, device_mesh):
        outputs = outputs.redistribute(placements=output_layouts)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, self._partition_fn, partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts), partial(self._prepare_output_fn, self.output_layouts, self.use_local_output))