import warnings
import collections
import json
import os
import re
from typing import List, Iterable
from qiskit import circuit
from qiskit.providers.models import BackendProperties, BackendConfiguration, PulseDefaults
from qiskit.providers import BackendV2, BackendV1
from qiskit import pulse
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers import basic_provider
from qiskit.transpiler import Target
from qiskit.providers.backend_compat import convert_to_target
from .utils.json_decoder import (
def _get_conf_dict_from_json(self):
    if not self.conf_filename:
        return None
    conf_dict = self._load_json(self.conf_filename)
    decode_backend_configuration(conf_dict)
    conf_dict['backend_name'] = self.backend_name
    return conf_dict