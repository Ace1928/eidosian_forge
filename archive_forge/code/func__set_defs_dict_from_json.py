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
def _set_defs_dict_from_json(self):
    if self.defs_filename:
        defs_dict = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs_dict)
        self._defs_dict = defs_dict