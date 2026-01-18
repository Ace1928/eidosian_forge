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
def _fuse_conv_bn_add_left(is_qat, add, bn_conv, _):
    bn, conv = bn_conv
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn, add)}')
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)