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
class BinaryUfuncInfo(OpInfo):
    """Operator information for 'universal binary functions (binary ufuncs).'
    These are functions of two tensors with common properties like:
      - they are elementwise functions
      - the output shape is determined by the input shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/stable/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """

    def __init__(self, name, *, sample_inputs_func=sample_inputs_elementwise_binary, reference_inputs_func=reference_inputs_elementwise_binary, error_inputs_func=None, lhs_make_tensor_kwargs=None, rhs_make_tensor_kwargs=None, always_returns_bool=False, supports_rhs_python_scalar=True, supports_one_python_scalar=False, supports_two_python_scalars=False, **kwargs):
        self._original_binary_ufunc_args = locals().copy()
        common_skips = (DecorateInfo(unittest.skip('Skipping redundant test.'), 'TestCommon', 'test_numpy_refs'),)
        kwargs['skips'] = kwargs.get('skips', tuple()) + common_skips
        super().__init__(name, sample_inputs_func=sample_inputs_func, reference_inputs_func=reference_inputs_func, error_inputs_func=make_error_inputs_elementwise_binary(error_inputs_func), **kwargs)
        if lhs_make_tensor_kwargs is None:
            lhs_make_tensor_kwargs = {}
        self.lhs_make_tensor_kwargs = lhs_make_tensor_kwargs
        if rhs_make_tensor_kwargs is None:
            rhs_make_tensor_kwargs = {}
        self.rhs_make_tensor_kwargs = rhs_make_tensor_kwargs
        self.always_returns_bool = always_returns_bool
        self.supports_rhs_python_scalar = supports_rhs_python_scalar
        self.supports_one_python_scalar = supports_one_python_scalar
        self.supports_two_python_scalars = supports_two_python_scalars
        if self.supports_two_python_scalars:
            self.supports_one_python_scalar = True
        if self.supports_one_python_scalar:
            assert supports_rhs_python_scalar, "Can't support lhs and rhs Python scalars but not rhs scalars!"