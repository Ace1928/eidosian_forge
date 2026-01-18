import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def _single_qubit_gates_arg_to_factory(single_qubit_gates: Sequence['cirq.Gate'], qubits: Sequence['cirq.Qid'], prng: 'np.random.RandomState') -> _SingleQubitLayerFactory:
    """Parse the `single_qubit_gates` argument for circuit generation functions.

    If only one single qubit gate is provided, it will be used everywhere.
    Otherwise, we use the factory that excludes operations that were used
    in the previous layer. This check is done by gate identity, not equality.
    """
    if len(set(single_qubit_gates)) == 1:
        return _FixedSingleQubitLayerFactory({q: single_qubit_gates[0] for q in qubits})
    return _RandomSingleQubitLayerFactory(qubits, single_qubit_gates, prng)