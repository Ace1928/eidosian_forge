import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def _observe_and_quantize_weight(weight):
    if dtype == torch.qint8:
        weight_observer = weight_observer_method()
        weight_observer(weight)
        qweight = _quantize_weight(weight.float(), weight_observer)
        return qweight
    else:
        return weight.float()