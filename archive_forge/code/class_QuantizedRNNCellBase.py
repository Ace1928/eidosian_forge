import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'scale_hh', 'scale_ih', 'zero_point_ih', 'zero_point_hh']

    def __init__(self, other):
        super().__init__()
        warnings.warn('torch.jit.QuantizedRNNCellBase is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.RNNCell instead.')
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.bias = other.bias
        if not self.bias:
            raise ValueError('Quantized RNN cells require bias terms')
        weight_ih, col_offsets_ih, self.scale_ih, self.zero_point_ih = torch.fbgemm_linear_quantize_weight(other.weight_ih.clone(memory_format=torch.contiguous_format).float())
        self.register_buffer('weight_ih', weight_ih)
        self.register_buffer('col_offsets_ih', col_offsets_ih)
        weight_hh, col_offsets_hh, self.scale_hh, self.zero_point_hh = torch.fbgemm_linear_quantize_weight(other.weight_hh.clone(memory_format=torch.contiguous_format).float())
        self.register_buffer('weight_hh', weight_hh)
        self.register_buffer('col_offsets_hh', col_offsets_hh)
        packed_ih = torch.fbgemm_pack_quantized_matrix(self.weight_ih)
        self.register_buffer('packed_ih', packed_ih)
        packed_hh = torch.fbgemm_pack_quantized_matrix(self.weight_hh)
        self.register_buffer('packed_hh', packed_hh)
        self.bias_ih = torch.nn.Parameter(other.bias_ih.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)
        self.bias_hh = torch.nn.Parameter(other.bias_hh.clone(memory_format=torch.contiguous_format).float(), requires_grad=False)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    @torch.jit.script_method
    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(f'input has inconsistent input_size: got {input.size(1)}, expected {self.input_size}')

    @torch.jit.script_method
    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str='') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(f"Input batch size {input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}")
        if hx.size(1) != self.hidden_size:
            raise RuntimeError(f'hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}')

    @torch.jit.script_method
    def _unpack(self):
        self.packed_ih.set_(torch.fbgemm_pack_quantized_matrix(self.weight_ih))
        self.packed_hh.set_(torch.fbgemm_pack_quantized_matrix(self.weight_hh))

    @torch.jit.script_method
    def _pack(self):
        self.packed_ih.set_(torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())
        self.packed_hh.set_(torch.zeros(torch.jit.annotate(List[int], []), dtype=torch.uint8).detach())