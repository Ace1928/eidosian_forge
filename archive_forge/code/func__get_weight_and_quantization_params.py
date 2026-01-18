import torch
import torch.nn as nn
from torch import Tensor
from .utils import _quantize_and_dequantize_weight
from .utils import _quantize_weight
from typing import Optional, Dict, Any, Tuple
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
def _get_weight_and_quantization_params(module, wn):
    weight = getattr(module, wn)
    params = [weight]
    for param_name in [wn + n for n in ['_qscheme', '_dtype', '_scale', '_zero_point', '_axis_int']]:
        if hasattr(module, param_name):
            param = getattr(module, param_name)
        else:
            param = None
        params.append(param)
    return params