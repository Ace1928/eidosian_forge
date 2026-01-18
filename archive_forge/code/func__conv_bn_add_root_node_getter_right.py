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
def _conv_bn_add_root_node_getter_right(pattern):
    add, _, bn_conv = pattern
    bn, conv = bn_conv
    return conv