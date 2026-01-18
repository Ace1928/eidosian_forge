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
def _square_observable(obs):
    """Returns the square of an observable."""
    if isinstance(obs, qml.operation.Tensor):
        components_squared = []
        for comp in obs.obs:
            try:
                components_squared.append(NONINVOLUTORY_OBS[comp.name](comp))
            except KeyError:
                pass
        return qml.operation.Tensor(*components_squared)
    return NONINVOLUTORY_OBS[obs.name](obs)