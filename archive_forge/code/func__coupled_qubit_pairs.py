import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def _coupled_qubit_pairs(qubits: Set['cirq.GridQubit']) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    pairs = []
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubits:
                pairs.append((qubit, neighbor))
        add_pair(cirq.GridQubit(qubit.row, qubit.col + 1))
        add_pair(cirq.GridQubit(qubit.row + 1, qubit.col))
    return pairs