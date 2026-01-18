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
class foreach_pointwise_sample_func(foreach_inputs_sample_func):

    def __init__(self, arity: int=3, rightmost_supports_scalar: bool=False, rightmost_supports_scalarlist: bool=False):
        super().__init__(arity, rightmost_supports_scalar, rightmost_supports_scalarlist)

    def _should_disable_fastpath(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        return dtype in integral_types_and(torch.bool) and opinfo.ref in (torch.addcmul,)

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs)
        args = [sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs) for _ in range(2)]
        if 'scalars' in kwargs:
            del kwargs['scalars']
        kwargs.update(self._sample_kwargs(opinfo, args[-1], ForeachRightmostArgType.TensorList, dtype))
        yield ForeachSampleInput(input, *args, **kwargs)

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        num_input_tensors_specified = 'num_input_tensors' in kwargs
        num_input_tensors = kwargs.pop('num_input_tensors') if num_input_tensors_specified else foreach_num_tensors
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for num_tensors, rightmost_arg_type in itertools.product(num_input_tensors, self._rightmost_arg_types):
            input = sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            args = [sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs) for _ in range(2 - int(rightmost_arg_type == ForeachRightmostArgType.TensorList))]
            rightmost_arg_list = self._sample_rightmost_arg(opinfo, rightmost_arg_type, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            for rightmost_arg in rightmost_arg_list:
                kwargs = {}
                if rightmost_arg_type == ForeachRightmostArgType.TensorList:
                    args.append(rightmost_arg)
                elif rightmost_arg_type in [ForeachRightmostArgType.Tensor, ForeachRightmostArgType.ScalarList]:
                    kwargs['scalars'] = rightmost_arg
                else:
                    kwargs['value'] = rightmost_arg
                kwargs.update(self._sample_kwargs(opinfo, rightmost_arg, rightmost_arg_type, dtype))
                assert len(args) == 2, f'len(args)={len(args)!r}'
                sample = ForeachSampleInput(input, *args, **kwargs)
                yield sample
                if rightmost_arg_type == ForeachRightmostArgType.TensorList:
                    args.pop()