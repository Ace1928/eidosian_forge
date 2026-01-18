from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals
def _generate_calibration_defaults(self) -> PulseDefaults:
    """Generate pulse calibration defaults as specified with `self._calibrate_instructions`.
        If `self._calibrate_instructions` is True, the pulse schedules will be generated from
        a series of default calibration sequences. If `self._calibrate_instructions` is False,
        the pulse schedules will contain empty calibration sequences, but still be generated and
        added to the target.
        """
    calibration_buffer = self._basis_gates.copy()
    for inst in ['delay', 'reset']:
        calibration_buffer.remove(inst)
    cmd_def = []
    for inst in calibration_buffer:
        num_qubits = self._supported_gates[inst].num_qubits
        qarg_set = self._coupling_map if num_qubits > 1 else list(range(self.num_qubits))
        if inst == 'measure':
            cmd_def.append(Command(name=inst, qubits=qarg_set, sequence=self._get_calibration_sequence(inst, num_qubits, qarg_set) if self._calibrate_instructions else []))
        else:
            for qarg in qarg_set:
                qubits = [qarg] if num_qubits == 1 else qarg
                cmd_def.append(Command(name=inst, qubits=qubits, sequence=self._get_calibration_sequence(inst, num_qubits, qubits) if self._calibrate_instructions else []))
    qubit_freq_est = np.random.normal(4.8, scale=0.01, size=self.num_qubits).tolist()
    meas_freq_est = np.linspace(6.4, 6.6, self.num_qubits).tolist()
    return PulseDefaults(qubit_freq_est=qubit_freq_est, meas_freq_est=meas_freq_est, buffer=0, pulse_library=_PULSE_LIBRARY, cmd_def=cmd_def)