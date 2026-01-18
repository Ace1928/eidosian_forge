from qiskit.exceptions import QiskitError
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from .fake_qasm_backend import FakeQasmBackend
from .utils.json_decoder import decode_pulse_defaults
def _set_defaults_from_json(self):
    if not self.props_filename:
        raise QiskitError('No properties file has been defined')
    defs = self._load_json(self.defs_filename)
    decode_pulse_defaults(defs)
    self._defaults = PulseDefaults.from_dict(defs)