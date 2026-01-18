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
def _preprocess_circuit(circuit: QuantumCircuit):
    num_bits_dict = {creg.name: creg.size for creg in circuit.cregs}
    mapping = _final_measurement_mapping(circuit)
    qargs = sorted(set(mapping.values()))
    qargs_index = {v: k for k, v in enumerate(qargs)}
    circuit = circuit.remove_final_measurements(inplace=False)
    if _has_control_flow(circuit):
        raise QiskitError('StatevectorSampler cannot handle ControlFlowOp')
    if _has_measure(circuit):
        raise QiskitError('StatevectorSampler cannot handle mid-circuit measurements')
    sentinel = len(qargs)
    indices = {key: [sentinel] * val for key, val in num_bits_dict.items()}
    for key, qreg in mapping.items():
        creg, ind = key
        indices[creg.name][ind] = qargs_index[qreg]
    meas_info = [_MeasureInfo(creg_name=name, num_bits=num_bits, num_bytes=_min_num_bytes(num_bits), qreg_indices=indices[name]) for name, num_bits in num_bits_dict.items()]
    return (circuit, qargs, meas_info)