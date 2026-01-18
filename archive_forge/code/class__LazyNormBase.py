from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from .. import functional as F
from .. import init
from ._functions import SyncBatchNorm as sync_batch_norm
from .lazy import LazyModuleMixin
from .module import Module
class _LazyNormBase(LazyModuleMixin, _NormBase):
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(0, eps, momentum, False, False, **factory_kwargs)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = UninitializedParameter(**factory_kwargs)
            self.bias = UninitializedParameter(**factory_kwargs)
        if self.track_running_stats:
            self.running_mean = UninitializedBuffer(**factory_kwargs)
            self.running_var = UninitializedBuffer(**factory_kwargs)
            self.num_batches_tracked = torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'})

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]
            if self.affine:
                assert isinstance(self.weight, UninitializedParameter)
                assert isinstance(self.bias, UninitializedParameter)
                self.weight.materialize((self.num_features,))
                self.bias.materialize((self.num_features,))
            if self.track_running_stats:
                self.running_mean.materialize((self.num_features,))
                self.running_var.materialize((self.num_features,))
            self.reset_parameters()