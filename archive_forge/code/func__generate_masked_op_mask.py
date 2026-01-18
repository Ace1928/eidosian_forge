import unittest
from collections.abc import Sequence
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy
def _generate_masked_op_mask(input_shape, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.bool, device=device, requires_grad=False)
    yield None
    yield make_arg(input_shape)
    if len(input_shape) > 2:
        yield make_arg(input_shape[:-1] + (1,))
        yield make_arg(input_shape[:1] + (1,) + input_shape[2:])
        yield make_arg((1,) + input_shape[1:])
        yield make_arg(input_shape[1:])
        yield make_arg(input_shape[-1:])