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
def _do_backward(self, gradients, retain_variables):
    self.retain_variables = retain_variables
    result = super()._do_backward(gradients, retain_variables)
    if not retain_variables:
        del self._nested_output
        del self._to_save_nested
    return result