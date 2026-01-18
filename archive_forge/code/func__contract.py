from typing import Sequence, Callable
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.pulse import ParametrizedEvolution, HardwareHamiltonian
from pennylane import transform
from .parameter_shift import _make_zero_rep
from .general_shift_rules import eigvals_to_frequencies, generate_shift_rule
from .gradient_transform import (
def _contract(coeffs, res, cjac):
    """Contract three tensors, the first two like a standard matrix multiplication
        and the result with the third tensor along the first axes."""
    return jnp.tensordot(jnp.tensordot(coeffs, res, axes=1), cjac, axes=[[0], [0]])