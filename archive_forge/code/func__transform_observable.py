from typing import Sequence, Callable
import itertools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import (
from .finite_difference import finite_diff
from .general_shift_rules import generate_shifted_tapes, process_shifts
from .gradient_transform import _no_trainable_grad
from .parameter_shift import _get_operation_recipe, expval_param_shift
def _transform_observable(obs, Z, device_wires):
    """Apply a Gaussian linear transformation to an observable.

    Args:
        obs (.Observable): observable to transform
        Z (array[float]): Heisenberg picture representation of the linear transformation
        device_wires (.Wires): wires on the device the transformed observable is to be
            measured on

    Returns:
        .Observable: the transformed observable
    """
    if obs.ev_order > 2:
        raise NotImplementedError('Transforming observables of order > 2 not implemented.')
    A = obs.heisenberg_obs(device_wires)
    if A.ndim != obs.ev_order:
        raise ValueError('Mismatch between the polynomial order of observable and its Heisenberg representation')
    A = A @ Z
    if A.ndim == 2:
        A = A + A.T
    return qml.PolyXP(A, wires=device_wires)