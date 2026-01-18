import json
import os
from qiskit.exceptions import QiskitError
from qiskit.providers.models import BackendProperties, QasmBackendConfiguration
from .utils.json_decoder import (
from .fake_backend import FakeBackend
def _get_conf_from_json(self):
    if not self.conf_filename:
        raise QiskitError('No configuration file has been defined')
    conf = self._load_json(self.conf_filename)
    decode_backend_configuration(conf)
    configuration = self._get_config_from_dict(conf)
    configuration.backend_name = self.backend_name
    return configuration