import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
def _do_forward(self, *input):
    self._nested_input = input
    flat_input = tuple(_iter_tensors(input))
    flat_output = super()._do_forward(*flat_input)
    nested_output = self._nested_output
    nested_tensors = _unflatten(flat_output, self._nested_output)
    return nested_tensors