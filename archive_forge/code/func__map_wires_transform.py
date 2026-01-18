from functools import partial
from typing import Callable, Union, Sequence
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
@partial(transform)
def _map_wires_transform(tape: qml.tape.QuantumTape, wire_map=None, queue=False) -> (Sequence[qml.tape.QuantumTape], Callable):
    ops = [map_wires(op, wire_map, queue=queue) if not isinstance(op, QuantumScript) else map_wires(op, wire_map, queue=queue)[0][0] for op in tape.operations]
    measurements = [map_wires(m, wire_map, queue=queue) for m in tape.measurements]
    out = tape.__class__(ops=ops, measurements=measurements, shots=tape.shots, trainable_params=tape.trainable_params)

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""
        return res[0]
    return ([out], processing_fn)