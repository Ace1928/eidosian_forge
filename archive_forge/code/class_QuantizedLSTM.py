import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedLSTM(QuantizedRNNBase):
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, other, dtype):
        super().__init__(other, dtype)
        warnings.warn('torch.jit.QuantizedLSTM is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.LSTM instead.')

    @torch.jit.script_method
    def forward_impl(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]], batch_sizes: Optional[Tensor], max_batch_size: int, sorted_indices: Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = self.permute_hidden(hx, sorted_indices)
        self.check_forward_args(input, hx, batch_sizes)
        assert batch_sizes is None
        result = torch.quantized_lstm(input, hx, self.all_weights, self.bias, self.num_layers, float(self.dropout), self.training, self.bidirectional, self.batch_first, dtype=self.dtype, use_dynamic=False)
        output = result[0]
        hidden = result[1:]
        return (output, hidden)

    @torch.jit.script_method
    def forward_tensor(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None
        output, hidden = self.forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        return (output, self.permute_hidden(hidden, unsorted_indices))

    @torch.jit.script_method
    def forward_packed(self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])
        output, hidden = self.forward_impl(input_, hx, batch_sizes, max_batch_size, sorted_indices)
        output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return (output, self.permute_hidden(hidden, unsorted_indices))

    @torch.jit.script_method
    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return (apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation))

    @torch.jit.script_method
    def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden[0], expected_hidden_size, 'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size, 'Expected hidden[1] size {}, got {}')

    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)