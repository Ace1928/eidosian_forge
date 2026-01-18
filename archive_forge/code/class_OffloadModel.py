from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
class OffloadModel(nn.Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train by offloading majority of the model parameters to the CPU.
    `OffloadModel` is heavily inspired by the _L2L algorithm and _Zero-Offload.
    ::

        model = get_model()
        offload_model = OffloadModel(model, device,
                                    offload_device=torch.device(“cpu”),
                                    num_slices=3,
                                    checkpoint_activation=True,
                                    num_microbatches=5)

    .. _L2L: https://arxiv.org/abs/2002.05645
    .. _Zero-Offload: https://arxiv.org/abs/2101.06840

    At each step, a layer(or series of layers) are loaded
    onto the GPU for the forward and backward pass with intermediate
    activations being copied onto the GPU as required. Once the forward
    or backward pass is completed for a given shard, it is moved back to
    the CPU again.

    `OffloadModel` supports activation checkpointing which reduces
    the memory footprint. You can also increase the number of
    microbatches which translates to more computation cycles for
    every shard load. This helps offset the cost of moving the shard
    from the CPU to GPU and vice versa.

    Note: OffloadModel currently only supports nn.Sequential models.

    Args:
        module (~torch.nn.Sequential): Module to be offloaded.

        device (torch.device):
            Device where the active model should reside.

        offload_device (torch.device):
            Device where the inactive model should reside.

        num_slices (int):
            Number of slices into which the model should be chunked.

        checkpoint_activation (bool):
            Boolean to indicate if we want to checkpoint intermediate
            activation states on the CPU. Default value is False.

        num_microbatches (int):
            Number of microbatches which should be run per model
            shard on device.
    """

    def __init__(self, model: Any, device: torch.device, offload_device: torch.device=torch.device('cpu'), num_slices: int=3, checkpoint_activation: bool=False, num_microbatches: int=1):
        super().__init__()
        if not model:
            raise TypeError('`model` argument to `OffloadModel` cannot be None.')
        if not device:
            raise TypeError('`device` argument to `OffloadModel` cannot be None.')
        if not (isinstance(model, nn.Sequential) or type(model) == list):
            raise TypeError('`model` argument to `OffloadModel` must be of type `nn.Sequential`.')
        if not torch.cuda.is_available():
            raise TypeError('CUDA must be available as one of the compute devices for `OffloadModel`.')
        self.device = device
        self.offload_device = offload_device
        self.model_slices: List[nn.Module] = []
        if type(model) == list:
            for i, m in enumerate(model):
                self.model_slices.append(ModelShard(cpu_model_shard=m, device=device, offload_device=offload_device, index=i))
        else:
            splits = _split(model, num_slices)
            for i, split in enumerate(splits):
                self.model_slices.append(ModelShard(cpu_model_shard=nn.Sequential(*split), device=device, offload_device=offload_device, index=i))
        self._model = torch.nn.Sequential(*self.model_slices)
        self._activations: List[Tuple] = []
        if not checkpoint_activation and num_microbatches > 1:
            raise RuntimeError('We currently only support microbatches with activation checkpointing.')
        self._checkpoint_activation = checkpoint_activation
        self._num_microbatches = num_microbatches

    def forward(self, *inputs: Any, **_: Any) -> Any:
        if self._checkpoint_activation:
            return OffloadFunction.apply(*inputs, torch.tensor([], requires_grad=True), self)
        self._activations = []
        for index in range(-1, len(self.model_slices)):
            if index >= 0:
                self._activations[index] = tuple([a.cuda() for a in list(self._activations[index])])
                inputs = self._activations[index]
                inputs = self.model_slices[index](*inputs)
            inputs = ShardSyncLayer.apply(inputs, index, self.model_slices, self)
            self._activations.append(inputs)
            if index >= 0:
                self._activations[index] = tuple([a.cpu() for a in list(self._activations[index])])
        result = self._activations[-1]
        result = tuple([r.cuda() for r in result])
        return result[0] if len(result) == 1 else result