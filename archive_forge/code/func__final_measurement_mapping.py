from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import warnings
import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QiskitError, QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.quantum_info import Statevector
from .base import BaseSamplerV2
from .base.validation import _has_measure
from .containers import (
from .containers.sampler_pub import SamplerPub
from .containers.bit_array import _min_num_bytes
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction
def _final_measurement_mapping(circuit: QuantumCircuit) -> dict[tuple[ClassicalRegister, int], int]:
    """Return the final measurement mapping for the circuit.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of classical bits to qubits for final measurements.
    """
    active_qubits = set(range(circuit.num_qubits))
    active_cbits = set(range(circuit.num_clbits))
    mapping = {}
    for item in circuit[::-1]:
        if item.operation.name == 'measure':
            loc = circuit.find_bit(item.clbits[0])
            cbit = loc.index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                for creg in loc.registers:
                    mapping[creg] = qbit
                active_cbits.remove(cbit)
        elif item.operation.name not in ['barrier', 'delay']:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)
        if not active_cbits or not active_qubits:
            break
    return mapping