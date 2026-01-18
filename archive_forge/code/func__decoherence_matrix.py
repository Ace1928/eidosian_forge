import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
def _decoherence_matrix(cool_rate: float, dephase_rate: float, heat_rate: float=0.0, dim: int=2) -> np.ndarray:
    """Construct a rate matrix associated with decay and dephasing.

    The units of the matrix match the units of the rates specified.
    This matrix can be used to construct a noise channel after rescaling
    by an idling time (to make it dimensionless).

    Args:
        cool_rate: Decay rate of the system, usually 1 / T_1
        dephase_rate: Static dephasing rate of the system, usually 1 / T_phi
        heat_rate: Heating rate of the system (default 0).
        dim: Number of energy levels to include (default 2).

    Returns:
        np.ndarray rate matrix for decay and dephasing.
    """
    rate_matrix = np.diag(np.arange(1, dim) * heat_rate, 1).T.astype(float)
    rate_matrix += np.diag(np.arange(1, dim) * cool_rate, 1)
    rate_matrix += np.diag(dephase_rate * np.arange(dim) ** 2)
    return rate_matrix