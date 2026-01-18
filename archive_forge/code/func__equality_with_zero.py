from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _equality_with_zero(context: cirq.DecompositionContext, qubits: Sequence[cirq.Qid], z: cirq.Qid) -> cirq.OP_TREE:
    if len(qubits) == 1:
        q, = qubits
        yield cirq.X(q)
        yield cirq.CNOT(q, z)
        return
    ancilla = context.qubit_manager.qalloc(len(qubits) - 2)
    yield and_gate.And(cv=[0] * len(qubits)).on(*qubits, *ancilla, z)