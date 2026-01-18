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
def _analytical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
    output = self._forward(module, input)
    output_size = output.nelement()
    if jacobian_input:
        jacobian_inp = self._jacobian(input, output_size)
        flat_jacobian_input = list(_iter_tensors(jacobian_inp))
    if jacobian_parameters:
        num_param = sum((p.numel() for p in self._get_parameters(module)[0]))
        jacobian_param = torch.zeros(num_param, output_size)
    for i in range(output_size):
        param, d_param = self._get_parameters(module)
        d_param = [torch.zeros_like(p) if d is None else d for p, d in zip(param, d_param)]
        d_out = torch.zeros_like(output)
        flat_d_out = d_out.view(-1)
        flat_d_out[i] = 1
        if jacobian_parameters:
            self._zero_grad_parameters(module)
        if jacobian_input:
            self._zero_grad_input(input)
        d_input = self._backward(module, input, output, d_out)
        if jacobian_input:
            for jacobian_x, d_x in zip(flat_jacobian_input, _iter_tensors(d_input)):
                jacobian_x[:, i] = d_x.contiguous().view(-1)
        if jacobian_parameters:
            jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)
    res: Tuple[torch.Tensor, ...] = tuple()
    if jacobian_input:
        res += (jacobian_inp,)
    if jacobian_parameters:
        res += (jacobian_param,)
    return res