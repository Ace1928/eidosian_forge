from typing import Sequence, Callable
import pennylane as qml
from pennylane.measurements import MidMeasureMP, ProbabilityMP, SampleMP, CountsMP, MeasurementValue
from pennylane.ops.op_math import ctrl
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from pennylane.queuing import QueuingManager
def _collect_mid_measure_info(tape: QuantumTape):
    """Helper function to collect information related to mid-circuit measurements in the tape."""
    measured_wires = []
    reused_measurement_wires = set()
    any_repeated_measurements = False
    is_postselecting = False
    for op in tape:
        if isinstance(op, MidMeasureMP):
            if op.postselect is not None:
                is_postselecting = True
            if op.reset:
                reused_measurement_wires.add(op.wires[0])
            if op.wires[0] in measured_wires:
                any_repeated_measurements = True
            measured_wires.append(op.wires[0])
        else:
            reused_measurement_wires = reused_measurement_wires.union(set(measured_wires).intersection(op.wires.toset()))
    return (measured_wires, reused_measurement_wires, any_repeated_measurements, is_postselecting)