from typing import List, TYPE_CHECKING
import functools
import numpy as np
from cirq import ops, protocols, qis, sim
def _Sdg(q: int, args: sim.CliffordTableauSimulationState, operations: List[ops.Operation], qubits: List['cirq.Qid']):
    protocols.act_on(ops.S ** (-1), args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.S(qubits[q]))