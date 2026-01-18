from collections import deque
import contextlib
import functools
from itertools import chain
import logging
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Union
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd.profiler as profiler
import torch.distributed as dist
from fairscale.internal.params import Workhandle, get_global_rank
from fairscale.nn.misc import GradBucket
from fairscale.optim import OSS
def _setup_backward_hooks(self) -> None:
    """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """
    with profiler.record_function('fairscale::sdp::setup_backward_hooks'):
        while len(self._grad_hooks) > 0:
            self._grad_hooks.pop().remove()
        self._grad_accs = []
        self._manual_reduce = []
        if not self.training:
            return
        for index, param in enumerate(self._trainable_params):
            if param.grad is not None and param.grad.requires_grad:
                raise RuntimeError("ShardedDataParallel only works with gradients that don't require grad")
            p_tmp = param.expand_as(param)
            if p_tmp.grad_fn is not None:
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                dst_rank = self._trainable_param_to_rank[param]
                reduce_function = self._get_reduce_fn(index, param, dst_rank)
                self._grad_hooks.append(grad_acc.register_hook(reduce_function))
                self._grad_accs.append(grad_acc)
                self._manual_reduce.append(reduce_function)