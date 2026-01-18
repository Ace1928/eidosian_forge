from qiskit.exceptions import QiskitError
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from .fake_qasm_backend import FakeQasmBackend
from .utils.json_decoder import decode_pulse_defaults
Returns a snapshot of device defaults