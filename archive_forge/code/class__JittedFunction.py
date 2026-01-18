import re
from typing import Callable, List
import torch
from torch import Tensor
class _JittedFunction:

    def __init__(self, code_string: str, return_by_ref: bool, num_outputs: int, **kwargs):
        self.code_string = code_string
        assert return_by_ref or num_outputs == 1, 'Return by value only works for single output. '
        self.return_by_ref = return_by_ref
        self.num_outputs = num_outputs
        parsed_code = _CodeParser(code_string)
        self.kernel_name = parsed_code.function_name
        self.kwargs_dict = kwargs
        self.is_cuda_available = torch.cuda.is_available()

    def __call__(self, *tensors: Tensor, **kwargs):
        assert self.is_cuda_available, 'Jiterator is only supported on CUDA and ROCm GPUs, none are available.'
        assert len(tensors) <= 8, 'jiterator only supports up to 8 tensor inputs.'
        expanded_kwargs = self.kwargs_dict.copy()
        for key, value in kwargs.items():
            if key in self.kwargs_dict:
                expanded_kwargs[key] = value
            else:
                raise KeyError(f'{key} is not declared in function definition')
        return torch._C._cuda_jiterator_compile_and_launch_kernel(self.code_string, self.kernel_name, self.return_by_ref, self.num_outputs, tensors, expanded_kwargs)