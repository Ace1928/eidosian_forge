from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
class foreach_inputs_sample_func:

    def __init__(self, arity: int, rightmost_supports_scalar: bool, rightmost_supports_scalarlist: bool, rightmost_supports_tensor: bool=False) -> None:
        self.arity = arity
        self._set_rightmost_arg_types(rightmost_supports_scalar, rightmost_supports_scalarlist, rightmost_supports_tensor)

    def _set_rightmost_arg_types(self, rightmost_supports_scalar: bool, rightmost_supports_scalarlist: bool, rightmost_supports_tensor: bool) -> None:
        self._rightmost_arg_types = [ForeachRightmostArgType.TensorList]
        if self.arity > 1:
            if rightmost_supports_scalar:
                self._rightmost_arg_types.append(ForeachRightmostArgType.Scalar)
            if rightmost_supports_scalarlist:
                self._rightmost_arg_types.append(ForeachRightmostArgType.ScalarList)
            if rightmost_supports_tensor:
                self._rightmost_arg_types.append(ForeachRightmostArgType.Tensor)

    def _sample_rightmost_arg(self, opinfo, rightmost_arg_type, device, dtype, num_tensors, **_foreach_inputs_kwargs):
        if rightmost_arg_type == ForeachRightmostArgType.TensorList:
            return [sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs)]
        if rightmost_arg_type == ForeachRightmostArgType.Tensor:
            return [make_tensor((), device=device, dtype=dtype, noncontiguous=_foreach_inputs_kwargs['noncontiguous'], requires_grad=_foreach_inputs_kwargs.get('requires_grad', False))]
        should_use_simpler_scalars = opinfo.name == '_foreach_pow' and dtype in (torch.float16, torch.bfloat16)

        def sample_float():
            s = random.random()
            if should_use_simpler_scalars:
                return 1.0 if s > 0.5 else 2.0
            else:
                return 1.0 - s
        high = 2 if should_use_simpler_scalars else 9
        if rightmost_arg_type == ForeachRightmostArgType.ScalarList:
            return [[random.randint(0, high) + 1 for _ in range(num_tensors)], [sample_float() for _ in range(num_tensors)], [complex(sample_float(), sample_float()) for _ in range(num_tensors)], [True for _ in range(num_tensors)], [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 3)], [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 4)]]
        if rightmost_arg_type == ForeachRightmostArgType.Scalar:
            return (random.randint(1, high + 1), sample_float(), True, complex(sample_float(), sample_float()))
        raise AssertionError(f'Invalid rightmost_arg_type of {rightmost_arg_type}')

    def _should_disable_fastpath(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        if self.arity == 1:
            if 'foreach_abs' in opinfo.name and dtype in complex_types():
                return True
            if opinfo.ref in (torch.abs, torch.neg):
                return False
            return dtype in integral_types_and(torch.bool)
        if self.arity < 2 or rightmost_arg_type == ForeachRightmostArgType.Tensor:
            return None
        if 'foreach_pow' in opinfo.name and dtype in integral_types():
            return True
        if rightmost_arg_type == ForeachRightmostArgType.TensorList:
            disable_fastpath = 'foreach_div' in opinfo.name and dtype in integral_types_and(torch.bool)
            if 'foreach_add' in opinfo.name and dtype == torch.bool:
                disable_fastpath = True
            return disable_fastpath
        elif rightmost_arg_type == ForeachRightmostArgType.Scalar:
            disable_fastpath = 'foreach_div' in opinfo.name and dtype in integral_types_and(torch.bool)
            if isinstance(rightmost_arg, bool):
                disable_fastpath |= dtype == torch.bool
                if opinfo.ref in (torch.add, torch.mul):
                    disable_fastpath = False
            elif isinstance(rightmost_arg, int):
                disable_fastpath |= dtype == torch.bool
            elif isinstance(rightmost_arg, float):
                disable_fastpath |= dtype in integral_types_and(torch.bool)
            elif isinstance(rightmost_arg, complex):
                disable_fastpath |= dtype not in complex_types()
            else:
                raise AssertionError(f'Invalid scalar of type {rightmost_arg_type} - {rightmost_arg}')
            return disable_fastpath
        elif rightmost_arg_type == ForeachRightmostArgType.ScalarList:
            disable_fastpath = opinfo.ref == torch.div and dtype in integral_types_and(torch.bool)
            elmt_t = type(rightmost_arg[0])
            has_same_type = all((isinstance(v, elmt_t) for v in rightmost_arg))
            if not has_same_type:
                return dtype not in complex_types()
            if isinstance(rightmost_arg[0], bool):
                if ('foreach_add' in opinfo.name or 'foreach_mul' in opinfo.name) and dtype == torch.bool:
                    disable_fastpath = False
            elif isinstance(rightmost_arg[0], int):
                disable_fastpath |= dtype == torch.bool
            elif isinstance(rightmost_arg[0], float):
                disable_fastpath |= dtype in integral_types_and(torch.bool)
            elif isinstance(rightmost_arg[0], complex):
                disable_fastpath |= dtype not in complex_types()
            else:
                raise AssertionError(f'Invalid scalarlist of {rightmost_arg}')
            return disable_fastpath
        else:
            raise AssertionError(f'Invalid rightmost_arg_type of {rightmost_arg_type}')

    def _sample_kwargs(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        kwargs = {}
        if rightmost_arg_type == ForeachRightmostArgType.TensorList and opinfo.supports_alpha_param:
            if dtype in integral_types_and(torch.bool):
                kwargs['alpha'] = 3
            elif dtype.is_complex:
                kwargs['alpha'] = complex(3, 3)
            else:
                kwargs['alpha'] = 3.14
        if self.arity > 1:
            kwargs['disable_fastpath'] = self._should_disable_fastpath(opinfo, rightmost_arg, rightmost_arg_type, dtype)
        return kwargs

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for rightmost_arg_type in self._rightmost_arg_types:
            zero_size_foreach_inputs_kwargs = copy.deepcopy(_foreach_inputs_kwargs)
            zero_size_foreach_inputs_kwargs['zero_size'] = True
            input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs)
            if self.arity > 1:
                args = [sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs) for _ in range(self.arity - 2)]
                args.append(self._sample_rightmost_arg(opinfo, ForeachRightmostArgType.TensorList, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs)[0])
                kwargs = self._sample_kwargs(opinfo, args[-1], ForeachRightmostArgType.TensorList, dtype, zero_size=True)
            else:
                args = []
                kwargs = {}
                if opinfo.ref in (torch.abs, torch.neg):
                    kwargs['disable_fastpath'] = False
                else:
                    kwargs['disable_fastpath'] = dtype in integral_types_and(torch.bool)
            yield ForeachSampleInput(input, *args, **kwargs)

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        num_input_tensors_specified = 'num_input_tensors' in kwargs
        num_input_tensors = kwargs.pop('num_input_tensors') if num_input_tensors_specified else foreach_num_tensors
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        _foreach_inputs_kwargs['zero_size'] = False
        for num_tensors, rightmost_arg_type, intersperse_empty_tensors in itertools.product(num_input_tensors, self._rightmost_arg_types, (True, False)):
            if intersperse_empty_tensors and (num_tensors != max(num_input_tensors) or str(device) == 'cpu'):
                continue
            _foreach_inputs_kwargs['intersperse_empty_tensors'] = intersperse_empty_tensors
            input = sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs)
            args = []
            if self.arity > 1:
                args = [sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs) for _ in range(self.arity - 2)]
                rightmost_arg_list = self._sample_rightmost_arg(opinfo, rightmost_arg_type, device, dtype, num_tensors, **_foreach_inputs_kwargs)
                for rightmost_arg in rightmost_arg_list:
                    args.append(rightmost_arg)
                    kwargs = self._sample_kwargs(opinfo, rightmost_arg, rightmost_arg_type, dtype)
                    ref_args = args
                    if rightmost_arg_type in (ForeachRightmostArgType.Scalar, ForeachRightmostArgType.Tensor):
                        ref_args = args[:-1] + [[args[-1] for _ in range(num_tensors)]]
                    sample = ForeachSampleInput(input, *args, ref_args=ref_args, **kwargs)
                    yield sample
                    args.pop()
            else:
                yield ForeachSampleInput(input, *args, disable_fastpath=self._should_disable_fastpath(opinfo, None, None, dtype))