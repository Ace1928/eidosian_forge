from __future__ import annotations
from typing import TYPE_CHECKING
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.exceptions import CircuitError
from ._builder_utils import validate_condition, condition_resources
from .control_flow import ControlFlowOp
class WhileLoopContext:
    """A context manager for building up while loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example.  All normal methods of accessing the qubits
    on the underlying :obj:`~QuantumCircuit` will work correctly, and resolve into correct accesses
    within the interior block.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.while_loop` in its context-manager form, i.e. by not supplying a ``body``
    or sets of qubits and clbits.

    Example usage::

        from qiskit.circuit import QuantumCircuit, Clbit, Qubit
        bits = [Qubit(), Qubit(), Clbit()]
        qc = QuantumCircuit(bits)

        with qc.while_loop((bits[2], 0)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_circuit', '_condition', '_label')

    def __init__(self, circuit: QuantumCircuit, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, *, label: str | None=None):
        self._circuit = circuit
        self._condition = validate_condition(condition)
        self._label = label

    def __enter__(self):
        resources = condition_resources(self._condition)
        self._circuit._push_scope(clbits=resources.clbits, registers=resources.cregs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        body = scope.build(scope.qubits(), scope.clbits())
        self._circuit.append(WhileLoopOp(self._condition, body, label=self._label), body.qubits, body.clbits)
        return False