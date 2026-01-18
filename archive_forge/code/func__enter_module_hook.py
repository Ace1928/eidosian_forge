import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def _enter_module_hook(self, name):

    class PopState(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *args):
            if len(args) == 1:
                return args[0]
            return args

        @staticmethod
        def backward(ctx, *grad_outs):
            self._exit_module(name)
            return grad_outs

    def f(module, inputs):
        self._enter_module(name)
        inputs = _normalize_tuple(inputs)
        out = PopState.apply(*inputs)
        return out
    return f