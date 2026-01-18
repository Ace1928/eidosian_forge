from typing import Optional, cast, TYPE_CHECKING, Iterable, Tuple, Dict
import sympy
import numpy as np
from cirq import circuits, ops, value, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _try_get_known_phased_pauli(op: ops.Operation, no_symbolic: bool=False) -> Optional[Tuple[value.TParamVal, value.TParamVal]]:
    if no_symbolic and protocols.is_parameterized(op):
        return None
    gate = op.gate
    if isinstance(gate, ops.PhasedXPowGate):
        e = gate.exponent
        p = gate.phase_exponent
    elif isinstance(gate, ops.YPowGate):
        e = gate.exponent
        p = 0.5
    elif isinstance(gate, ops.XPowGate):
        e = gate.exponent
        p = 0.0
    elif isinstance(gate, ops.PhasedXZGate) and (not protocols.is_parameterized(gate.z_exponent)) and np.isclose(float(gate.z_exponent), 0):
        e = gate.x_exponent
        p = gate.axis_phase_exponent
    else:
        return None
    return (value.canonicalize_half_turns(e), value.canonicalize_half_turns(p))