from __future__ import annotations
from typing import TYPE_CHECKING
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.exceptions import CircuitError
from ._builder_utils import validate_condition, condition_resources
from .control_flow import ControlFlowOp
class WhileLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit (``body``) until
    a condition (``condition``) evaluates as False.

    Parameters:
        condition: A condition to be checked prior to executing ``body``. Can be
            specified as either a tuple of a ``ClassicalRegister`` to be tested
            for equality with a given ``int``, or as a tuple of a ``Clbit`` to
            be compared to either a ``bool`` or an ``int``.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    The classical bits used in ``condition`` must be a subset of those attached
    to ``body``.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────┐
        q_0: ┤0            ├
             │             │
        q_1: ┤1            ├
             │  while_loop │
        q_2: ┤2            ├
             │             │
        c_0: ╡0            ╞
             └─────────────┘

    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, body: QuantumCircuit, label: str | None=None):
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits
        super().__init__('while_loop', num_qubits, num_clbits, [body], label=label)
        self.condition = validate_condition(condition)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        from qiskit.circuit import QuantumCircuit
        body, = parameters
        if not isinstance(body, QuantumCircuit):
            raise CircuitError(f'WhileLoopOp expects a body parameter of type QuantumCircuit, but received {type(body)}.')
        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(f'Attempted to assign a body parameter with a num_qubits or num_clbits different than that of the WhileLoopOp. WhileLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}.')
        self._params = [body]

    @property
    def blocks(self):
        return (self._params[0],)

    def replace_blocks(self, blocks):
        body, = blocks
        return WhileLoopOp(self.condition, body, label=self.label)

    def c_if(self, classical, val):
        raise NotImplementedError('WhileLoopOp cannot be classically controlled through Instruction.c_if. Please use an IfElseOp instead.')