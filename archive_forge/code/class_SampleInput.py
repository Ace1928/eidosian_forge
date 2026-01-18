import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
class SampleInput:
    """Represents sample inputs to a function."""
    __slots__ = ['input', 'args', 'kwargs', 'output_process_fn_grad', 'broadcasts_input', 'name']

    def __init__(self, input, *var_args, args=None, kwargs=None, output_process_fn_grad=None, broadcasts_input=None, name=None, **var_kwargs):
        self.input = input
        if args is not None or kwargs is not None:
            assert not var_args and (not var_kwargs), '\nA SampleInput can be constructed "naturally" with *args and **kwargs or by\nexplicitly setting the "args" and "kwargs" parameters, but the two\nmethods of construction cannot be mixed!'
        elif len(var_args) or len(var_kwargs):
            assert output_process_fn_grad is None and broadcasts_input is None and (name is None), '\nA SampleInput constructed "naturally" with *args and **kwargs\ncannot specify additional metadata in keyword arguments'
        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)
        self.output_process_fn_grad = output_process_fn_grad if output_process_fn_grad is not None else lambda x: x
        self.name = name if name is not None else ''
        self.broadcasts_input = broadcasts_input if broadcasts_input is not None else False

    def with_metadata(self, *, output_process_fn_grad=None, broadcasts_input=None, name=None):
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        if name is not None:
            self.name = name
        return self

    def _repr_helper(self, formatter):
        arguments = [f'input={formatter(self.input)}', f'args={formatter(self.args)}', f'kwargs={formatter(self.kwargs)}', f'broadcasts_input={self.broadcasts_input}', f'name={repr(self.name)}']
        return f'SampleInput({', '.join((a for a in arguments if a is not None))})'

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):

        def formatter(arg):
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ''
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and (not arg.is_contiguous()):
                    contiguity_suffix = ', contiguous=False'
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return 'TensorList[' + ', '.join(map(formatter, arg)) + ']'
            elif isinstance(arg, (list, tuple)):
                return '(' + ','.join(map(formatter, arg)) + ')'
            return repr(arg)
        return self._repr_helper(formatter)

    def transform(self, f):

        def tt(t):

            def _tt(t):
                with torch.no_grad():
                    return f(t)
            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t
        sample_tt_input, tt_args, tt_kwargs = (tt(self.input), tt(self.args), tt(self.kwargs))
        return SampleInput(sample_tt_input, args=tt_args, kwargs=tt_kwargs, output_process_fn_grad=self.output_process_fn_grad, broadcasts_input=self.broadcasts_input, name=self.name + '_transformed')

    def numpy(self):

        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]
            return t
        return self.transform(to_numpy)

    def noncontiguous(self):

        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t
            return t
        return self.transform(to_noncontiguous)