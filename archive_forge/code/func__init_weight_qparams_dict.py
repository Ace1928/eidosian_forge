import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def _init_weight_qparams_dict(self, weight_qparams_dict, device):
    self.is_decomposed = weight_qparams_dict['is_decomposed']
    for key, weight_qparams in weight_qparams_dict.items():
        if key == 'is_decomposed':
            continue
        weight_qscheme = weight_qparams['qscheme']
        weight_dtype = weight_qparams['dtype']
        setattr(self, key + '_qscheme', weight_qscheme)
        setattr(self, key + '_dtype', weight_dtype)
        assert weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], Exception(f'qscheme: {weight_qscheme} is not support in {self._get_name()}')
        if weight_qscheme is not None:
            self.register_buffer(key + '_scale', torch.tensor(weight_qparams['scale'], dtype=torch.float, device=device))
            self.register_buffer(key + '_zero_point', torch.tensor(weight_qparams['zero_point'], dtype=torch.int, device=device))
            if weight_qscheme == torch.per_channel_affine:
                self.register_buffer(key + '_axis', torch.tensor(weight_qparams['axis'], dtype=torch.int, device=device))
            else:
                self.register_buffer(key + '_axis', torch.tensor(0, dtype=torch.int, device=device))
            setattr(self, key + '_axis_int', getattr(self, key + '_axis').item())