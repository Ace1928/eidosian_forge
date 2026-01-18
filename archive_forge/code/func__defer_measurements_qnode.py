from typing import Sequence, Callable
import pennylane as qml
from pennylane.measurements import MidMeasureMP, ProbabilityMP, SampleMP, CountsMP, MeasurementValue
from pennylane.ops.op_math import ctrl
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from pennylane.queuing import QueuingManager
@defer_measurements.custom_qnode_transform
def _defer_measurements_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``defer_measurements``."""
    if tkwargs.get('device', None):
        raise ValueError("Cannot provide a 'device' value directly to the defer_measurements decorator when transforming a QNode.")
    tkwargs.setdefault('device', qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)