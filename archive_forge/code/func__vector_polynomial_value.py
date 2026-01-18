from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _vector_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) vector input `x`.
    Check out `_polynomial_value` function for more details.
    """

    def transition(curr_poly_val, x, poly_coeff):
        res = torch.addcmul(poly_coeff.unsqueeze(-1), x, curr_poly_val)
        return res
    if zero_power is None:
        zero_power = x.new_ones(1).expand(x.shape)
    return _polynomial_value(poly, x, zero_power, transition)