import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedGRU(QuantizedRNNBase):
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('torch.jit.QuantizedGRU is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.GRU instead.')

    @torch.jit.script_method
    def forward_impl(self, input: Tensor, hx: Optional[Tensor], batch_sizes: Optional[Tensor], max_batch_size: int, sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = self.permute_hidden(hx, sorted_indices)
        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = torch.quantized_gru(input, hx, self.all_weights, self.bias, self.num_layers, float(self.dropout), self.training, self.bidirectional, self.batch_first)
        else:
            result = torch.quantized_gru(input, batch_sizes, hx, self.all_weights, self.bias, self.num_layers, float(self.dropout), self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]
        return (output, hidden)

    @torch.jit.script_method
    def forward_tensor(self, input: Tensor, hx: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None
        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        return (output, self.permute_hidden(hidden, unsorted_indices))

    @torch.jit.script_method
    def forward_packed(self, input: PackedSequence, hx: Optional[Tensor]=None) -> Tuple[PackedSequence, Tensor]:
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])
        output, hidden = self.forward_impl(input_, hx, batch_sizes, max_batch_size, sorted_indices)
        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return (output, self.permute_hidden(hidden, unsorted_indices))

    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)