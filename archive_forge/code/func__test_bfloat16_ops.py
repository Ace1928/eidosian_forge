from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
def _test_bfloat16_ops(test_case, op, device, inp_dims=(), prec=0.01, scale_factor=None):
    input1 = torch.randn(inp_dims, dtype=torch.float32, device=device, requires_grad=True)
    if scale_factor is not None:
        input1 = (torch.rand(inp_dims, dtype=torch.bfloat16, device=device) * scale_factor).float().requires_grad_()
    out1 = op(input1)
    grad_input1 = torch.randn_like(out1, device=device)
    out1.backward(grad_input1)
    op_bfp16 = op.bfloat16()
    input2 = input1.detach().bfloat16().requires_grad_()
    grad_input2 = grad_input1.bfloat16()
    out2 = op_bfp16(input2)
    out2.backward(grad_input2)
    test_case.assertEqual(out1, out2, atol=prec, rtol=prec, exact_dtype=False)
    test_case.assertEqual(input1.grad.data, input2.grad.data, atol=prec, rtol=prec, exact_dtype=False)