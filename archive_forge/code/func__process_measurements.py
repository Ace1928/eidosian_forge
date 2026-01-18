from functools import partial
from typing import List, Union, Sequence, Callable
import networkx as nx
import pennylane as qml
from pennylane.transforms import transform
from pennylane import Hamiltonian
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.ops.qubit import SWAP
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape
def _process_measurements(expanded_tape, device_wires, is_default_mixed):
    measurements = expanded_tape.measurements.copy()
    if device_wires:
        for i, m in enumerate(measurements):
            if isinstance(m, qml.measurements.StateMP):
                if is_default_mixed:
                    measurements[i] = qml.density_matrix(wires=device_wires)
            elif not m.wires:
                measurements[i] = type(m)(wires=device_wires)
    return measurements