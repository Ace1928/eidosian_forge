from collections import deque
from contextlib import contextmanager
import threading
from typing import (
import torch
from torch import Tensor
import torch.autograd
from .dependency import fork, join
from .microbatch import Batch
from .phony import get_phony
class Recompute(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Context, phony: Tensor, recomputed: Deque[Recomputed], rng_states: Deque[RNGStates], function: Function, input_atomic: bool, *inputs) -> Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states
        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.inputs = inputs
        if input_atomic:
            tensors = [inputs[0]]
        else:
            tensors = []
            for input in inputs:
                if torch.is_tensor(input):
                    tensors.append(input)
        ctx.save_for_backward(*tensors)
        return phony

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor) -> Tuple[None, ...]:
        inputs = ctx.inputs
        inputs_leaf = tuple((x.detach().requires_grad_(x.requires_grad) if torch.is_tensor(x) else x for x in inputs))
        device = None
        for input in inputs:
            if torch.is_tensor(input):
                device = input.device
                break
        if device is None:
            raise RuntimeError(f'No tensors found in {inputs}')
        with restore_rng_states(device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                if ctx.input_atomic:
                    assert len(inputs_leaf) == 1
                    output = ctx.function(inputs_leaf[0])
                else:
                    output = ctx.function(*inputs_leaf)
        ctx.recomputed.append((output, inputs_leaf))
        grad_input: List[None] = [None, None, None, None, None]
        grad_input.extend((None for _ in ctx.inputs))
        return tuple(grad_input)