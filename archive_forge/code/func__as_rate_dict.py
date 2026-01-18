import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import sympy
from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
def _as_rate_dict(rate_or_dict: Optional[Union[float, Dict['cirq.Qid', float]]], qubits: Set['cirq.Qid']) -> Dict['cirq.Qid', float]:
    """Convert float or None input into dictionary form.

    This method also ensures that no qubits are missing from dictionary keys.
    """
    if rate_or_dict is None:
        return {q: 0.0 for q in qubits}
    elif isinstance(rate_or_dict, dict):
        return {**{q: 0.0 for q in qubits}, **rate_or_dict}
    else:
        return {q: rate_or_dict for q in qubits}