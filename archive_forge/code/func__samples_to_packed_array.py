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
def _samples_to_packed_array(samples: NDArray[np.uint8], num_bits: int, indices: list[int]) -> NDArray[np.uint8]:
    ary = np.pad(samples[:, ::-1], ((0, 0), (0, 1)), constant_values=0)
    ary = ary[:, indices[::-1]]
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    ary = np.packbits(ary, axis=-1)
    return ary