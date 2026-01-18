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
def _test_batched_grad(input, output, output_idx) -> bool:
    diff_input_list = list(_iter_tensors(input, True))
    grad = functools.partial(torch.autograd.grad, output, diff_input_list, retain_graph=True, allow_unused=True)

    def vjp(v):
        results = grad(v)
        results = tuple((grad if grad is not None else torch.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape) for grad, inp in zip(results, diff_input_list)))
        return results
    grad_outputs = [torch.randn_like(output) for _ in range(2)]
    expected = [vjp(gO) for gO in grad_outputs]
    expected = [torch.stack(shards) for shards in zip(*expected)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='There is a performance drop')
        warnings.filterwarnings('ignore', message='Please use torch.vmap')
        try:
            result = vmap(vjp)(torch.stack(grad_outputs))
        except RuntimeError as ex:
            raise GradcheckError(f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}') from ex
    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        raise GradcheckError(_get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))
    return True