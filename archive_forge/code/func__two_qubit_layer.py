import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def _two_qubit_layer(coupled_qubit_pairs: List[GridQubitPairT], two_qubit_op_factory: Callable[['cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'], 'cirq.OP_TREE'], layer: GridInteractionLayer, prng: 'np.random.RandomState') -> 'cirq.OP_TREE':
    for a, b in coupled_qubit_pairs:
        if (a, b) in layer or (b, a) in layer:
            yield two_qubit_op_factory(a, b, prng)