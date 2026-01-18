from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _polynomial_value(poly, x, zero_power, transition):
    """
    A generic method for computing poly(x) using the Horner's rule.

    Args:
      poly (Tensor): the (possibly batched) 1D Tensor representing
                     polynomial coefficients such that
                     poly[..., i] = (a_{i_0}, ..., a{i_n} (==1)), and
                     poly(x) = poly[..., 0] * zero_power + ... + poly[..., n] * x^n

      x (Tensor): the value (possible batched) to evalate the polynomial `poly` at.

      zero_power (Tensor): the representation of `x^0`. It is application-specific.

      transition (Callable): the function that accepts some intermediate result `int_val`,
                             the `x` and a specific polynomial coefficient
                             `poly[..., k]` for some iteration `k`.
                             It basically performs one iteration of the Horner's rule
                             defined as `x * int_val + poly[..., k] * zero_power`.
                             Note that `zero_power` is not a parameter,
                             because the step `+ poly[..., k] * zero_power` depends on `x`,
                             whether it is a vector, a matrix, or something else, so this
                             functionality is delegated to the user.
    """
    res = zero_power.clone()
    for k in range(poly.size(-1) - 2, -1, -1):
        res = transition(res, x, poly[..., k])
    return res