from __future__ import annotations
from collections.abc import Sequence
import cmath
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .ucrz import UCRZGate
def _extract_rz(phi1, phi2):
    """
    Extract a Rz rotation (angle given by first output) such that exp(j*phase)*Rz(z_angle)
    is equal to the diagonal matrix with entires exp(1j*ph1) and exp(1j*ph2).
    """
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return (phase, z_angle)