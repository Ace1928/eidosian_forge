from typing import Optional
import torch
from torch.ao.nn.quantized.modules.utils import _quantize_weight, _hide_packed_params_repr
class LinearPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, row_block_size=1, col_block_size=4, dtype=torch.qint8):
        super().__init__()
        if dtype != torch.qint8:
            raise NotImplementedError('Linear prepacking only supports QINT8')
        self.dtype = dtype
        wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        self.set_weight_bias(wq, None, row_block_size, col_block_size)

    def _get_name(self):
        return 'SparseQuantizedLinearPackedParams'

    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor], row_block_size: Optional[int], col_block_size: Optional[int]) -> None:
        assert row_block_size is not None and col_block_size is not None
        self._packed_params = torch.ops.sparse.qlinear_prepack(weight, bias, row_block_size, col_block_size)

    @torch.jit.export
    def _weight_bias(self):
        weight, bias, block_sizes = torch.ops.sparse.qlinear_unpack(self._packed_params)
        return (weight, bias, block_sizes[0], block_sizes[1])

    def forward(self, x):
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + '_packed_params'] = self._weight_bias()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        assert version <= self._version
        self.dtype = state_dict.pop(prefix + 'dtype')
        weight, bias, row_block_size, col_block_size = state_dict.pop(prefix + '_packed_params')
        self.set_weight_bias(weight, bias, row_block_size, col_block_size)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    @torch.jit.export
    def __getstate__(self):
        return (self._packed_params, self.training, self.dtype)

    @torch.jit.export
    def __setstate__(self, state):
        self._packed_params, self.training, self.dtype = state

    def __repr__(self):
        return self._weight_bias().__repr__()