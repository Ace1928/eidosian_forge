from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
def _multi_meas_grad(res, coeffs, r0, unshifted_coeff, num_measurements):
    """Compute the gradient for multiple measurements by taking the linear combination of
    the coefficients and each measurement result."""
    g = []
    if r0 is None:
        r0 = [None] * num_measurements
    for meas_idx in range(num_measurements):
        meas_result = [param_result[meas_idx] for param_result in res]
        g_component = _single_meas_grad(meas_result, coeffs, unshifted_coeff, r0[meas_idx])
        g.append(g_component)
    return tuple(g)