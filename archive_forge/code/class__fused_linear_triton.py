import math
from typing import Any, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.components.activations import Activation
from xformers.triton.k_activations import get_triton_activation_index
from xformers.triton.k_fused_matmul_bw import fused_matmul_backward
from xformers.triton.k_fused_matmul_fw import fused_matmul
class _fused_linear_triton(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, weight, bias, activation, trainable_weight, trainable_bias, save_activation_inputs):
        y, activation_inputs = fused_matmul(x, weight, bias, activation, save_activation_inputs)
        ctx.activation = activation
        ctx.trainable_weight = trainable_weight
        ctx.trainable_bias = trainable_bias
        if x.requires_grad or ctx.trainable_weight or ctx.trainable_bias:
            ctx.save_for_backward(weight, activation_inputs, x)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad_out: torch.Tensor) -> Any:
        """
        Compute the derivative with respect to x, other tensors were not trainable inputs.
        """
        weight, activation_inputs, x = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_matmul_backward(grad_out=grad_out, inputs=x, act_in=activation_inputs, weight=weight, trainable_weight=ctx.trainable_weight, trainable_bias=ctx.trainable_bias, activation_grad=ctx.activation)
        return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)