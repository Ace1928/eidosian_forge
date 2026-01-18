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
def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    """Given the linear, bn and leaky_relu modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
                or post training quantization fusion
        linear: Module instance of type Linear
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer
    Examples::
        >>> # xdoctest: +SKIP(failing)
        >>> m1 = nn.Linear(20, 10)
        >>> b1 = nn.BatchNorm1d(10)
        >>> lr = nn.LeakyReLU(0.01)
        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)
    """
    assert linear.training == bn.training and bn.training == leaky_relu.training, 'Linear, BN and LeakyReLU all must be in the same mode (train or eval).'
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(linear, bn, leaky_relu)}')
    else:
        map_to_fused_module_eval = {nn.Linear: nni.LinearLeakyReLU}
        fused_module = map_to_fused_module_eval.get(type(linear), None)
        if fused_module is not None:
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            fm = fused_module(fused_linear, leaky_relu)
            return fm
        else:
            raise NotImplementedError(f'Cannot fuse eval modules: {(linear, bn, leaky_relu)}')