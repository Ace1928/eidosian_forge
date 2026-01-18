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
def _forward_slow(self, input):
    """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """
    assert self.bn.running_var is not None
    assert self.bn.running_mean is not None
    zero_bias = torch.zeros(self.out_channels, device=self.weight.device, dtype=input.dtype)
    weight_shape = [1] * len(self.weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(self.weight.shape)
    bias_shape[1] = -1
    if self.bn.training:
        conv_out = self._conv_forward(input, self.weight, zero_bias)
        with torch.no_grad():
            conv_out_bias = conv_out if self.bias is None else conv_out + self.bias.reshape(bias_shape)
            self.bn(conv_out_bias)
    running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
    scale_factor = self.bn.weight / running_std
    scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
    conv_bn = self._conv_forward(input, scaled_weight, zero_bias)
    if self.bn.training:
        avg_dims = [0] + list(range(2, len(self.weight.shape)))
        batch_mean = conv_out.mean(avg_dims)
        batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(avg_dims)
        batch_std = torch.sqrt(batch_var + self.bn.eps)
        unscale_factor = running_std / batch_std
        conv_bn *= unscale_factor.reshape(bias_shape)
        fused_mean = batch_mean
        fused_std = batch_std
    else:
        fused_mean = self.bn.running_mean - (self.bias if self.bias is not None else 0)
        fused_std = running_std
    fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
    conv_bn += fused_bias.reshape(bias_shape)
    if self.bias is not None:
        conv_bn += (self.bias - self.bias).reshape(bias_shape)
    return conv_bn