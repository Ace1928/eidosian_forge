from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit
from .piecewise_linear_pauli_rotations import PiecewiseLinearPauliRotations
def _check_sizes_match(slope, offset, breakpoints):
    size = len(slope)
    if len(offset) != size:
        raise ValueError(f'Size mismatch of slope ({size}) and offset ({len(offset)}).')
    if breakpoints is not None:
        if len(breakpoints) != size:
            raise ValueError(f'Size mismatch of slope ({size}) and breakpoints ({len(breakpoints)}).')