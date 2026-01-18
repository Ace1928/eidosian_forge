import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
@functools.lru_cache(maxsize=256)
def _kraus_ops_from_rates(flat_rates: Tuple[float, ...], shape: Tuple[int, int]) -> Sequence[np.ndarray]:
    """Generate kraus operators from an array of rates.

    Args:
        flat_rates: A tuple of rates, flattened from a numpy array with:
            flat_rates = tuple(rates.reshape(-1))
            This format is necessary to support caching of inputs.
        shape: The shape of flat_rates prior to flattening.
    """
    rates = np.array(flat_rates).reshape(shape)
    num_op = np.diag(np.sqrt(np.diag(rates)))
    annihilation = np.sqrt(np.triu(rates, 1))
    creation = np.sqrt(np.triu(rates.T, 1)).T
    L = _lindbladian(annihilation) + _lindbladian(creation) + 2 * _lindbladian(num_op)
    superop = linalg.expm(L.real)
    return qis.superoperator_to_kraus(superop)