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
class NestedIOFunction(Function):

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_tensors(input))
        flat_output = super()._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_tensors = _unflatten(flat_output, self._nested_output)
        return nested_tensors

    def _do_backward(self, gradients, retain_variables):
        self.retain_variables = retain_variables
        result = super()._do_backward(gradients, retain_variables)
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        return result

    def backward(self, *gradients: Any) -> Any:
        nested_gradients = _unflatten(gradients, self._nested_output)
        result = self.backward_extended(*nested_gradients)
        return tuple(_iter_None_tensors(result))
    __call__ = _do_forward

    def forward(self, *args: Any) -> Any:
        nested_tensors = _map_tensor_data(self._nested_input)
        result = self.forward_extended(*nested_tensors)
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))

    def save_for_backward(self, *args: Any) -> None:
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    @property
    def saved_tensors(self):
        flat_tensors = super().saved_tensors
        return _unflatten(flat_tensors, self._to_save_nested)

    def mark_dirty(self, *args: Any, **kwargs: Any) -> None:
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args: Any, **kwargs: Any) -> None:
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    def forward_extended(self, *input: Any) -> None:
        raise NotImplementedError

    def backward_extended(self, *grad_output: Any) -> None:
        raise NotImplementedError