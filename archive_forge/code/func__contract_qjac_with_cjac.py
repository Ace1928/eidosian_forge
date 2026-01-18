import itertools as it
import warnings
from functools import partial
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ProbabilityMP, StateMP, VarianceMP
from pennylane.transforms import transform
from .general_shift_rules import (
from .gradient_transform import find_and_validate_gradient_methods
from .parameter_shift import _get_operation_recipe
from .hessian_transform import _process_jacs
def _contract_qjac_with_cjac(qhess, cjac, tape):
    """Contract a quantum Jacobian with a classical preprocessing Jacobian."""
    if len(tape.measurements) > 1:
        qhess = qhess[0]
    has_single_arg = False
    if not isinstance(cjac, tuple):
        has_single_arg = True
        cjac = (cjac,)
    hessians = []
    for jac in cjac:
        if jac is not None:
            hess = _process_jacs(jac, qhess)
            hessians.append(hess)
    return hessians[0] if has_single_arg else tuple(hessians)