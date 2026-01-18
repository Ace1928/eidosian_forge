from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models.pulsedefaults import PulseDefaults
Returns a dictionary of `qiskit.providers.backend.QubitProperties` using
    a backend properties dictionary created by loading props.json payload.
    