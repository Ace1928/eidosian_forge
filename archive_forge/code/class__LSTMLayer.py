import numbers
from typing import Optional, Tuple
import warnings
import torch
from torch import Tensor
class _LSTMLayer(torch.nn.Module):
    """A single bi-directional LSTM layer."""

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool=True, batch_first: bool=False, bidirectional: bool=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _LSTMSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)
        if self.bidirectional:
            self.layer_bw = _LSTMSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]]=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        if hidden is None:
            hx_fw, cx_fw = (None, None)
        else:
            hx_fw, cx_fw = hidden
        hidden_bw: Optional[Tuple[Tensor, Tensor]] = None
        if self.bidirectional:
            if hx_fw is None:
                hx_bw = None
            else:
                hx_bw = hx_fw[1]
                hx_fw = hx_fw[0]
            if cx_fw is None:
                cx_bw = None
            else:
                cx_bw = cx_fw[1]
                cx_fw = cx_fw[0]
            if hx_bw is not None and cx_bw is not None:
                hidden_bw = (hx_bw, cx_bw)
        if hx_fw is None and cx_fw is None:
            hidden_fw = None
        else:
            hidden_fw = (torch.jit._unwrap_optional(hx_fw), torch.jit._unwrap_optional(cx_fw))
        result_fw, hidden_fw = self.layer_fw(x, hidden_fw)
        if hasattr(self, 'layer_bw') and self.bidirectional:
            x_reversed = x.flip(0)
            result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
            result_bw = result_bw.flip(0)
            result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
            if hidden_fw is None and hidden_bw is None:
                h = None
                c = None
            elif hidden_fw is None:
                h, c = torch.jit._unwrap_optional(hidden_bw)
            elif hidden_bw is None:
                h, c = torch.jit._unwrap_optional(hidden_fw)
            else:
                h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)
                c = torch.stack([hidden_fw[1], hidden_bw[1]], 0)
        else:
            result = result_fw
            h, c = torch.jit._unwrap_optional(hidden_fw)
        if self.batch_first:
            result.transpose_(0, 1)
        return (result, (h, c))

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        """
        There is no FP equivalent of this class. This function is here just to
        mimic the behavior of the `prepare` within the `torch.ao.quantization`
        flow.
        """
        assert hasattr(other, 'qconfig') or qconfig is not None
        input_size = kwargs.get('input_size', other.input_size)
        hidden_size = kwargs.get('hidden_size', other.hidden_size)
        bias = kwargs.get('bias', other.bias)
        batch_first = kwargs.get('batch_first', other.batch_first)
        bidirectional = kwargs.get('bidirectional', other.bidirectional)
        layer = cls(input_size, hidden_size, bias, batch_first, bidirectional)
        layer.qconfig = getattr(other, 'qconfig', qconfig)
        wi = getattr(other, f'weight_ih_l{layer_idx}')
        wh = getattr(other, f'weight_hh_l{layer_idx}')
        bi = getattr(other, f'bias_ih_l{layer_idx}', None)
        bh = getattr(other, f'bias_hh_l{layer_idx}', None)
        layer.layer_fw = _LSTMSingleLayer.from_params(wi, wh, bi, bh)
        if other.bidirectional:
            wi = getattr(other, f'weight_ih_l{layer_idx}_reverse')
            wh = getattr(other, f'weight_hh_l{layer_idx}_reverse')
            bi = getattr(other, f'bias_ih_l{layer_idx}_reverse', None)
            bh = getattr(other, f'bias_hh_l{layer_idx}_reverse', None)
            layer.layer_bw = _LSTMSingleLayer.from_params(wi, wh, bi, bh)
        return layer