import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobjExperimentHeader(QobjDictField):
    """A header for a single OpenQASM 2 experiment in the qobj."""
    pass