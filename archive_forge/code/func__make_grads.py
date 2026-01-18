import warnings
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union
import torch
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge
from .. import _vmap_internals
from ..overrides import handle_torch_function, has_torch_function, is_tensor_like
from . import forward_ad, functional, graph
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .function import Function, NestedIOFunction
from .grad_mode import (
from .gradcheck import gradcheck, gradgradcheck
from .variable import Variable
from torch._C._autograd import (
from torch._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState
from . import profiler
def _make_grads(outputs: Sequence[torch.Tensor], grads: Sequence[_OptionalTensor], is_grads_batched: bool) -> Tuple[_OptionalTensor, ...]:
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            from torch.fx.experimental.symbolic_shapes import expect_true, sym_eq
            first_grad = grad if not is_grads_batched else grad[0]
            if out.is_nested or first_grad.is_nested:
                shape_matches = torch.is_same_size(out, first_grad)
            else:
                shape_matches = expect_true(sym_eq(out.size(), first_grad.size()))
            if not shape_matches:
                out_shape, grad_shape = _calculate_shape(out, first_grad, is_grads_batched)
                if is_grads_batched:
                    raise RuntimeError('If `is_grads_batched=True`, we interpret the first dimension of each grad_output as the batch dimension. The sizes of the remaining dimensions are expected to match the shape of corresponding output, but a mismatch was detected: grad_output[' + str(grads.index(grad)) + '] has a shape of ' + str(grad_shape) + ' and output[' + str(outputs.index(out)) + '] has a shape of ' + str(out_shape) + '. If you only want some tensors in `grad_output` to be considered batched, consider using vmap.')
                else:
                    raise RuntimeError('Mismatch in shape: grad_output[' + str(grads.index(grad)) + '] has a shape of ' + str(grad_shape) + ' and output[' + str(outputs.index(out)) + '] has a shape of ' + str(out_shape) + '.')
            if out.dtype.is_complex != grad.dtype.is_complex:
                raise RuntimeError('For complex Tensors, both grad_output and output are required to have the same dtype. Mismatch in dtype: grad_output[' + str(grads.index(grad)) + '] has a dtype of ' + str(grad.dtype) + ' and output[' + str(outputs.index(out)) + '] has a dtype of ' + str(out.dtype) + '.')
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError('grad can be implicitly created only for scalar outputs')
                if not out.dtype.is_floating_point:
                    msg = f'grad can be implicitly created only for real scalar outputs but got {out.dtype}'
                    raise RuntimeError(msg)
                new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
            else:
                new_grads.append(None)
        else:
            raise TypeError('gradients can be either Tensors or None, but got ' + type(grad).__name__)
    return tuple(new_grads)