from typing import Sequence, Callable
import pennylane as qml
from pennylane.measurements import MidMeasureMP, ProbabilityMP, SampleMP, CountsMP, MeasurementValue
from pennylane.ops.op_math import ctrl
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from pennylane.queuing import QueuingManager
def _add_control_gate(op, control_wires):
    """Helper function to add control gates"""
    control = [control_wires[m.id] for m in op.meas_val.measurements]
    new_ops = []
    for branch, value in op.meas_val._items():
        if value:
            qscript = qml.tape.make_qscript(ctrl(lambda: qml.apply(op.then_op), control=Wires(control), control_values=branch))()
            new_ops.extend(qscript.circuit)
    return new_ops