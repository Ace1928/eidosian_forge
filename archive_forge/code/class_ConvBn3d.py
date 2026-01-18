import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import TypeVar
class ConvBn3d(_ConvBnNd, nn.Conv3d):
    """
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = nni.ConvBn3d
    _FLOAT_CONV_MODULE = nn.Conv3d
    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE = None

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        _ConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig, dim=3)