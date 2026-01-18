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
def _check_circuits_coupling_map(circuits, cmap, backend):
    max_qubits = None
    if cmap is not None:
        max_qubits = cmap.size()
    elif backend is not None:
        backend_version = getattr(backend, 'version', 0)
        if backend_version <= 1:
            if not backend.configuration().simulator:
                max_qubits = backend.configuration().n_qubits
            else:
                max_qubits = None
        else:
            max_qubits = backend.num_qubits
    for circuit in circuits:
        num_qubits = len(circuit.qubits)
        if max_qubits is not None and num_qubits > max_qubits:
            raise CircuitTooWideForTarget(f'Number of qubits ({num_qubits}) in {circuit.name} is greater than maximum ({max_qubits}) in the coupling_map')