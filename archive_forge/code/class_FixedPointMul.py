import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging
class FixedPointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on hardware.

    Args:
        pre_act (`torch.Tensor`):
            Input tensor.
        pre_act_scaling_factor (`torch.Tensor`):
            Scaling factor of the input tensor *pre_act*.
        bit_num (`int`):
            Quantization bitwidth.
        z_scaling_factor (`torch.Tensor`):
            Scaling factor of the output tensor.
        identity (`torch.Tensor`, *optional*):
            Identity tensor, if exists.
        identity_scaling_factor (`torch.Tensor`, *optional*):
            Scaling factor of the identity tensor *identity*, if exists.

    Returns:
        `torch.Tensor`: Output tensor(*pre_act* if *identity* is not given, otherwise the addition of *pre_act* and
        *identity*), whose scale is rescaled to *z_scaling_factor*.
    """

    @staticmethod
    def forward(ctx, pre_act, pre_act_scaling_factor, bit_num, z_scaling_factor, identity=None, identity_scaling_factor=None):
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x: x
        else:
            reshape = lambda x: x.view(1, 1, -1)
        ctx.identity = identity
        n = 2 ** (bit_num - 1) - 1
        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)
            ctx.z_scaling_factor = z_scaling_factor
            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = z_scaling_factor.type(torch.float).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)
            m, e = batch_frexp(new_scale)
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / 2.0 ** e)
            if identity is not None:
                wx_int = torch.round(identity / identity_scaling_factor)
                _A = identity_scaling_factor.type(torch.double)
                _B = z_scaling_factor.type(torch.float).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)
                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / 2.0 ** e1)
                output = output1 + output
            return torch.clamp(output.type(torch.float), -n - 1, n)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return (grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None)