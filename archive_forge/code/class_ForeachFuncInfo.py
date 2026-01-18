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
class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions"""

    def __init__(self, name, sample_inputs_func, *, dtypes=floating_and_complex_types(), dtypesIfCUDA=floating_and_complex_types_and(torch.half), dtypesIfROCM=None, supports_alpha_param=False, supports_autograd=True, supports_inplace_autograd=True, supports_scalar_self_arg=False, supports_forward_ad=True, backward_requires_result=False, supports_out=True, **kwargs):
        foreach_method, foreach_method_inplace, torch_ref_method, torch_ref_inplace = get_foreach_method_names(name)
        if not supports_out:
            assert foreach_method is None
            assert torch_ref_method is None
            foreach_method = foreach_method_inplace
            torch_ref_method = torch_ref_inplace
        super().__init__(name='_foreach_' + name, op=foreach_method, ref=torch_ref_method, method_variant=foreach_method, inplace_variant=foreach_method_inplace, dtypes=dtypes, dtypesIfCUDA=dtypesIfCUDA, dtypesIfROCM=dtypesIfROCM, sample_inputs_func=sample_inputs_func, supports_autograd=supports_autograd, supports_forward_ad=supports_forward_ad, supports_out=supports_out, **kwargs)
        self.supports_scalar_self_arg = supports_scalar_self_arg
        self.ref_inplace = torch_ref_inplace
        self.supports_alpha_param = supports_alpha_param
        self.backward_requires_result = backward_requires_result
        self.has_no_in_place = self.inplace_variant is None
        self.supports_inplace_autograd = supports_inplace_autograd
        if name == 'norm':
            self.ref = torch.linalg.vector_norm
        elif name == 'minimum':
            self.ref = torch.clamp_max
            self.ref_inplace = torch.Tensor.clamp_max_
        elif name == 'maximum':
            self.ref = torch.clamp_min
            self.ref_inplace = torch.Tensor.clamp_min_

    def sample_zero_size_inputs(self, device, dtype, requires_grad=False, **kwargs):
        if not hasattr(self.sample_inputs_func, 'sample_zero_size_tensor_inputs'):
            return []
        return self.sample_inputs_func.sample_zero_size_tensor_inputs(self, device, dtype, requires_grad, **kwargs)