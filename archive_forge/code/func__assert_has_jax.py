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
def _assert_has_jax(transform_name):
    """Check that JAX is installed and imported correctly, otherwise raise an error.

    Args:
        transform_name (str): Name of the gradient transform that queries the return system
    """
    if not has_jax:
        raise ImportError(f'Module jax is required for the {transform_name} gradient transform. You can install jax via: pip install jax jaxlib')