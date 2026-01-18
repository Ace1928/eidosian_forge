import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError('no Tensors requiring grad found in input')
    grads_input = torch.autograd.grad(outputs, diff_input_list, [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in outputs], allow_unused=True)
    for gi, di in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
            if gi.layout != di.layout:
                raise GradcheckError('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(di.layout) + ')')
            if _is_sparse_any_tensor(gi):
                sparse_kind = str(gi.layout).replace('torch.', '').replace('_coo', '')
                if gi.sparse_dim() != di.sparse_dim():
                    raise GradcheckError(f'grad is {sparse_kind} tensor, but has incorrect sparse_dim {gi.sparse_dim()}, expected {di.sparse_dim()}')
                if gi.dense_dim() != di.dense_dim():
                    raise GradcheckError(f'grad is {sparse_kind} tensor, but has incorrect dense_dim {gi.dense_dim()}, expected {di.dense_dim()}')
            gi = gi.to_dense()
            di = di.to_dense()
        if masked:
            if not torch.allclose(gi, torch.zeros_like(gi)):
                raise GradcheckError('backward not multiplied by grad_output')
        elif not gi.eq(0).all():
            raise GradcheckError('backward not multiplied by grad_output')
        if gi.dtype != di.dtype:
            raise GradcheckError('grad is incorrect type')
        if gi.device != di.device:
            raise GradcheckError('grad is incorrect device')
        if gi.size() != di.size():
            raise GradcheckError('grad is incorrect size')
    return True