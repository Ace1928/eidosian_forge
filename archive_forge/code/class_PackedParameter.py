import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
class PackedParameter(torch.nn.Module):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'param'] = self.param

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict[prefix + 'param']
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)