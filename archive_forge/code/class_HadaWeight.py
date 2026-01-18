import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
class HadaWeight(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = w1a @ w1b * (w2a @ w2b) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        w1a, w1b, w2a, w2b, scale = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp
        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp
        del temp
        return (grad_w1a, grad_w1b, grad_w2a, grad_w2b, None)