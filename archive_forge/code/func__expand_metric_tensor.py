from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
def _expand_metric_tensor(tape: qml.tape.QuantumTape, argnum=None, approx=None, allow_nonunitary=True, aux_wire=None, device_wires=None) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Set the metric tensor based on whether non-unitary gates are allowed."""
    if not allow_nonunitary and approx is None:
        return ([qml.transforms.expand_nonunitary_gen(tape)], lambda x: x[0])
    return ([qml.transforms.expand_multipar(tape)], lambda x: x[0])