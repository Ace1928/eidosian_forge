import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class FastGeLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp