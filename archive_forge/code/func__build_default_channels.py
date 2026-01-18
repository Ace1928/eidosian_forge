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
def _build_default_channels(self) -> None:
    channels_map = {'acquire': {(i,): [pulse.AcquireChannel(i)] for i in range(self.num_qubits)}, 'drive': {(i,): [pulse.DriveChannel(i)] for i in range(self.num_qubits)}, 'measure': {(i,): [pulse.MeasureChannel(i)] for i in range(self.num_qubits)}, 'control': {edge: [pulse.ControlChannel(i)] for i, edge in enumerate(self._coupling_map)}}
    setattr(self, 'channels_map', channels_map)