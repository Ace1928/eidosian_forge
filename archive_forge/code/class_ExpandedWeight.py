from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
class ExpandedWeight(torch.Tensor):

    def __init__(self, orig_weight, batch_size, loss_reduction):
        self.batch_size = batch_size
        self.batch_first = True
        self.allow_smaller_batches = False
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction
    handled_functions = HANDLED_FUNCTIONS

    def __new__(cls, orig_weight, batch_size, loss_reduction):
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f'Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}')
        if not orig_weight.requires_grad:
            raise RuntimeError('Can only build ExpandedWeights objects of tensors that require_grad')
        ret = torch.Tensor._make_subclass(cls, orig_weight, True)
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in expanded_weights_rnn_decomps:
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = isinstance(args[2], list)
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]
            if decomp is not None:
                with setup_rnn(use_input_variant, args, kwargs):
                    return decomp(*args, **kwargs)
        if func == torch._cudnn_rnn_flatten_weight:
            return
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(tuple(kwargs.keys()), func, *args + tuple(kwargs.values()))
        raise RuntimeError(f'Expanded Weights encountered but cannot handle function {func.__name__}')

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def data(self):
        return self.orig_weight.data

    @property
    def shape(self):
        return self.orig_weight.shape

    @property
    def device(self):
        return self.orig_weight.device

    @property
    def is_cuda(self):
        return self.orig_weight.is_cuda

    def data_ptr(self):
        return self.orig_weight.data_ptr()

    def get_device(self):
        return self.orig_weight.get_device()

    def set_allow_smaller_batches(self, is_allow_smaller_batches):
        self.allow_smaller_batches = is_allow_smaller_batches

    def set_batch_first(self, is_batch_first=True):
        self.batch_first = is_batch_first