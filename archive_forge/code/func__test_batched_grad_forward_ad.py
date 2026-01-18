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
def _test_batched_grad_forward_ad(func, inputs) -> bool:
    fwAD = torch.autograd.forward_ad
    assert isinstance(inputs, tuple)
    for input_idx, current_input in enumerate(inputs):
        if not (is_tensor_like(current_input) and current_input.requires_grad):
            continue

        def jvp(tangent: torch.Tensor):
            with fwAD.dual_level():
                dual = fwAD.make_dual(current_input.detach(), tangent)
                inputs_with_dual = tuple((dual if idx == input_idx else inp.detach() if is_tensor_like(inp) else inp for idx, inp in enumerate(inputs)))
                dual_outputs = _as_tuple(func(*inputs_with_dual))
                ret = []
                for dual_output in dual_outputs:
                    if dual_output is None:
                        continue
                    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
                    if tangent_out is not None:
                        ret.append(tangent_out)
                    else:
                        ret.append(torch.zeros([], dtype=primal_out.dtype, device=primal_out.device).expand(primal_out.shape))
                return tuple(ret)
        if not _is_float_or_complex_tensor(current_input):
            continue
        tangents = [torch.randn_like(current_input) for _ in range(2)]
        expected = [jvp(t) for t in tangents]
        expected = [torch.stack(shards) for shards in zip(*expected)]
        try:
            result = _vmap(jvp)(torch.stack(tangents))
        except RuntimeError as ex:
            raise GradcheckError(f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG_FWD_AD}') from ex
        for input_idx, (res, exp) in enumerate(zip(result, expected)):
            if torch.allclose(res, exp):
                continue
            raise GradcheckError(_get_failed_batched_grad_test_msg(input_idx, input_idx, res, exp, is_forward_ad=True))
    return True