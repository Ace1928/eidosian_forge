import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
import torch.ao.nn.quantized.reference as nnqr
from ._common_operator_config_utils import (
from .backend_config import (
from ..fuser_method_mappings import (
import operator
from torch.ao.quantization.utils import MatchAllNode
import itertools
def _conv_bn_add_extra_inputs_getter_right(pattern):
    """ get inputs pattern for extra inputs, inputs for root node
    are assumed to be copied over from root node to the fused node
    """
    _, extra_input, bn_conv = pattern
    bn, conv = bn_conv
    return [extra_input]