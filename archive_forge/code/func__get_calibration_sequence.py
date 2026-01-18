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
def _get_calibration_sequence(self, inst: str, num_qubits: int, qargs: tuple[int]) -> list[PulseQobjInstruction]:
    """Return calibration pulse sequence for given instruction (defined by name and num_qubits)
        acting on qargs.
        """
    pulse_library = _PULSE_LIBRARY
    if inst == 'measure':
        sequence = [PulseQobjInstruction(name='acquire', duration=1792, t0=0, qubits=qargs, memory_slot=qargs)] + [PulseQobjInstruction(name=pulse_library[1].name, ch=f'm{i}', t0=0) for i in qargs]
        return sequence
    if num_qubits == 1:
        return [PulseQobjInstruction(name='fc', ch=f'u{qargs[0]}', t0=0, phase='-P0'), PulseQobjInstruction(name=pulse_library[0].name, ch=f'd{qargs[0]}', t0=0)]
    return [PulseQobjInstruction(name=pulse_library[1].name, ch=f'd{qargs[0]}', t0=0), PulseQobjInstruction(name=pulse_library[2].name, ch=f'u{qargs[0]}', t0=0), PulseQobjInstruction(name=pulse_library[1].name, ch=f'd{qargs[1]}', t0=0), PulseQobjInstruction(name='fc', ch=f'd{qargs[1]}', t0=0, phase=2.1)]