import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedGRUCell(QuantizedRNNCellBase):

    def __init__(self, other):
        super().__init__(other)
        warnings.warn('torch.jit.QuantizedGRUCell is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic.GRUCell instead.')

    @torch.jit.script_method
    def forward(self, input: Tensor, hx: Optional[Tensor]=None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        return _VF.quantized_gru_cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.packed_ih, self.packed_hh, self.col_offsets_ih, self.col_offsets_hh, self.scale_ih, self.scale_hh, self.zero_point_ih, self.zero_point_hh)