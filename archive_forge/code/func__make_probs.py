from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@transform
def _make_probs(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Ignores the return types of the provided circuit and creates a new one
    that outputs probabilities"""
    qscript = qml.tape.QuantumScript(tape.operations, [qml.probs(tape.wires)], shots=tape.shots)

    def post_processing_fn(res):
        return res[0]
    return ([qscript], post_processing_fn)