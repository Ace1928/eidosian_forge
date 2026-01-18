import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
def _validate_rates(qubits: Set['cirq.Qid'], rates: Dict['cirq.Qid', np.ndarray]) -> None:
    """Check all rate matrices are square and of appropriate dimension.

    We check rates are positive in the class validator.
    """
    if qubits != set(rates):
        raise ValueError('qubits for rates inconsistent with those through qubit_dims')
    for q in rates:
        if rates[q].shape != (q.dimension, q.dimension):
            raise ValueError(f'Invalid shape for rate matrix: should be ({q.dimension}, {q.dimension}), but got {rates[q].shape}')