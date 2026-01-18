from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane.transforms import transform
from pennylane.wires import Wires
def _add_operations_to_graph(tape, graph, gate_types, q_mapper, c_mapper):
    """Add the tape operation to the PyZX graph."""
    for op in tape.operations:
        name = op.name
        if name not in gate_types:
            raise qml.QuantumFunctionError('The expansion of the quantum tape failed, PyZX does not support', name)
        map_gate = gate_types[name]
        args = [*op.wires, *(p / np.pi for p in op.parameters)]
        gate = map_gate(*args)
        gate.to_graph(graph, q_mapper, c_mapper)