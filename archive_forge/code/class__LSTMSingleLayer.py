import numbers
from typing import Optional, Tuple
import warnings
import torch
from torch import Tensor
class _LSTMSingleLayer(torch.nn.Module):
    """A single one-directional LSTM layer.

    The difference between a layer and a cell is that the layer can process a
    sequence, while the cell only expects an instantaneous value.
    """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]]=None):
        result = []
        seq_len = x.shape[0]
        for i in range(seq_len):
            hidden = self.cell(x[i], hidden)
            result.append(hidden[0])
        result_tensor = torch.stack(result, 0)
        return (result_tensor, hidden)

    @classmethod
    def from_params(cls, *args, **kwargs):
        cell = LSTMCell.from_params(*args, **kwargs)
        layer = cls(cell.input_size, cell.hidden_size, cell.bias)
        layer.cell = cell
        return layer