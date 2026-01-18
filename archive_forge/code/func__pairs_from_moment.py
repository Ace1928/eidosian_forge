import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def _pairs_from_moment(moment: 'cirq.Moment') -> List[QidPairT]:
    """Helper function in `get_random_combinations_for_layer_circuit` pair generator.

    The moment should contain only two qubit operations, which define a list of qubit pairs.
    """
    pairs: List[QidPairT] = []
    for op in moment.operations:
        if len(op.qubits) != 2:
            raise ValueError('Layer circuit contains non-2-qubit operations.')
        qpair = cast(QidPairT, op.qubits)
        pairs.append(qpair)
    return pairs