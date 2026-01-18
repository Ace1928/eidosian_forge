from typing import List, Optional, Sequence, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import circuits, protocols, study, devices, ops, value
from cirq._doc import document
from cirq.sim import sparse_simulator, density_matrix_simulator
from cirq.sim.clifford import clifford_simulator
from cirq.transformers import measurement_transformers
def _to_circuit(program: 'cirq.CIRCUIT_LIKE') -> 'cirq.Circuit':
    if isinstance(program, circuits.Circuit):
        result = program
    elif isinstance(program, ops.Gate):
        result = circuits.Circuit(program.on(*devices.LineQid.for_gate(program)))
    else:
        result = circuits.Circuit(program)
    return result