import copy
import logging
from time import time
from typing import List, Union, Dict, Callable, Any, Optional, TypeVar
import warnings
from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.providers.models import BackendProperties
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.instruction_durations import InstructionDurations, InstructionDurationsType
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.target import Target, target_to_backend_properties
def _parse_instruction_durations(backend, inst_durations, dt, circuit):
    """Create a list of ``InstructionDuration``s. If ``inst_durations`` is provided,
    the backend will be ignored, otherwise, the durations will be populated from the
    backend. If any circuits have gate calibrations, those calibration durations would
    take precedence over backend durations, but be superceded by ``inst_duration``s.
    """
    if not inst_durations:
        backend_version = getattr(backend, 'version', 0)
        if backend_version <= 1:
            backend_durations = InstructionDurations()
            try:
                backend_durations = InstructionDurations.from_backend(backend)
            except AttributeError:
                pass
        else:
            backend_durations = backend.instruction_durations
    circ_durations = InstructionDurations()
    if not inst_durations:
        circ_durations.update(backend_durations, dt or backend_durations.dt)
    if circuit.calibrations:
        cal_durations = []
        for gate, gate_cals in circuit.calibrations.items():
            for (qubits, parameters), schedule in gate_cals.items():
                cal_durations.append((gate, qubits, parameters, schedule.duration))
        circ_durations.update(cal_durations, circ_durations.dt)
    if inst_durations:
        circ_durations.update(inst_durations, dt or getattr(inst_durations, 'dt', None))
    return circ_durations