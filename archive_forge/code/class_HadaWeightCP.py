import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
class HadaWeightCP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)
        rebuild1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
        rebuild2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)
        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        t1, w1a, w1b, t2, w2a, w2b, scale = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w2a)
        grad_w = rebuild * grad_out
        del rebuild
        grad_w1a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w1a.T)
        del grad_w, temp
        grad_w1b = torch.einsum('i r k l, i j k l -> r j', t1, grad_temp)
        grad_t1 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w1b.T)
        del grad_temp
        temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w1a)
        grad_w = rebuild * grad_out
        del rebuild
        grad_w2a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w2a.T)
        del grad_w, temp
        grad_w2b = torch.einsum('i r k l, i j k l -> r j', t2, grad_temp)
        grad_t2 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w2b.T)
        del grad_temp
        return (grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None)