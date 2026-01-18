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
def _get_noise_defaults(self, name: str, num_qubits: int) -> tuple:
    """Return noise default values/ranges for duration and error of supported
        instructions. There are two possible formats:
            - (min_duration, max_duration, min_error, max_error),
              if the defaults are ranges.
            - (duration, error), if the defaults are fixed values.
        """
    if name in _NOISE_DEFAULTS:
        return _NOISE_DEFAULTS[name]
    if num_qubits == 1:
        return _NOISE_DEFAULTS_FALLBACK['1-q']
    return _NOISE_DEFAULTS_FALLBACK['multi-q']