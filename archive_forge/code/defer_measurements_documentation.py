from typing import Sequence, Callable
import pennylane as qml
from pennylane.measurements import MidMeasureMP, ProbabilityMP, SampleMP, CountsMP, MeasurementValue
from pennylane.ops.op_math import ctrl
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from pennylane.queuing import QueuingManager
Helper function to add control gates