from typing import Optional
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, CircuitInstruction
from ..blueprintcircuit import BlueprintCircuit
def _warn_if_precision_loss(self):
    """Issue a warning if constructing the circuit will lose precision.

        If we need an angle smaller than ``pi * 2**-1022``, we start to lose precision by going into
        the subnormal numbers.  We won't lose _all_ precision until an exponent of about 1075, but
        beyond 1022 we're using fractional bits to represent leading zeros."""
    max_num_entanglements = self.num_qubits - self.approximation_degree - 1
    if max_num_entanglements > -np.finfo(float).minexp:
        warnings.warn(f'precision loss in QFT. The rotation needed to represent {max_num_entanglements} entanglements is smaller than the smallest normal floating-point number.', category=RuntimeWarning, stacklevel=3)