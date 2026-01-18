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
def _get_noise_model_from_backend_v2(self, gate_error=True, readout_error=True, thermal_relaxation=True, temperature=0, gate_lengths=None, gate_length_units='ns'):
    """Build noise model from BackendV2.

        This is a temporary fix until qiskit-aer supports building noise model
        from a BackendV2 object.
        """
    from qiskit.circuit import Delay
    from qiskit.providers.exceptions import BackendPropertyError
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device.models import _excited_population, basic_device_gate_errors, basic_device_readout_errors
    from qiskit_aer.noise.passes import RelaxationNoisePass
    if self._props_dict is None:
        self._set_props_dict_from_json()
    properties = BackendProperties.from_dict(self._props_dict)
    basis_gates = self.operation_names
    num_qubits = self.num_qubits
    dt = self.dt
    noise_model = NoiseModel(basis_gates=basis_gates)
    if readout_error:
        for qubits, error in basic_device_readout_errors(properties):
            noise_model.add_readout_error(error, qubits)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='qiskit_aer.noise.device.models')
        gate_errors = basic_device_gate_errors(properties, gate_error=gate_error, thermal_relaxation=thermal_relaxation, gate_lengths=gate_lengths, gate_length_units=gate_length_units, temperature=temperature)
    for name, qubits, error in gate_errors:
        noise_model.add_quantum_error(error, name, qubits)
    if thermal_relaxation:
        try:
            excited_state_populations = [_excited_population(freq=properties.frequency(q), temperature=temperature) for q in range(num_qubits)]
        except BackendPropertyError:
            excited_state_populations = None
        try:
            delay_pass = RelaxationNoisePass(t1s=[properties.t1(q) for q in range(num_qubits)], t2s=[properties.t2(q) for q in range(num_qubits)], dt=dt, op_types=Delay, excited_state_populations=excited_state_populations)
            noise_model._custom_noise_passes.append(delay_pass)
        except BackendPropertyError:
            pass
    return noise_model