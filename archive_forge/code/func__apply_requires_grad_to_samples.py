import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def _apply_requires_grad_to_samples(sample_inputs):
    """Decorator to _maybe_failing_sample_inputs_... generator functions
    that clones and sets requires_grad argument to tensors in sample
    input arguments. This is needed when the generated samples share
    tensor instances.
    """

    def wrapper(op_info, device, dtype, requires_grad, layout, **kwargs):

        def apply_requires_grad(x):
            if not isinstance(x, torch.Tensor) or x.requires_grad or (not requires_grad) or (not (x.is_floating_point() or x.is_complex())):
                return x
            return x.detach().clone().requires_grad_(requires_grad)
        if requires_grad:
            for sample_input in sample_inputs(op_info, device, dtype, requires_grad, layout, **kwargs):
                yield sample_input.transform(apply_requires_grad)
        else:
            yield from sample_inputs(op_info, device, dtype, requires_grad, layout, **kwargs)
    return wrapper