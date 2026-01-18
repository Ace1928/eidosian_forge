from typing import List, TYPE_CHECKING
import functools
import numpy as np
from cirq import ops, protocols, qis, sim
def _X(q: int, args: sim.CliffordTableauSimulationState, operations: List[ops.Operation], qubits: List['cirq.Qid']):
    protocols.act_on(ops.X, args, qubits=[qubits[q]], allow_decompose=False)
    operations.append(ops.X(qubits[q]))