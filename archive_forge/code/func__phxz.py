from typing import List
import cirq
def _phxz(a: float, x: float, z: float):
    return cirq.PhasedXZGate(axis_phase_exponent=a, x_exponent=x, z_exponent=z)