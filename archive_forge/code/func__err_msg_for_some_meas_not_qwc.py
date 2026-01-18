import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def _err_msg_for_some_meas_not_qwc(measurements):
    """Error message for the case when some operators measured on the same wire are not qubit-wise commuting."""
    return f'Only observables that are qubit-wise commuting Pauli words can be returned on the same wire, some of the following measurements do not commute:\n{measurements}'