from __future__ import annotations
import warnings
from typing import Iterable, Optional, Union, TYPE_CHECKING
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from .control_flow import ControlFlowOp
class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    Parameters:
        indexset: A collection of integers to loop over.
        loop_parameter: The placeholder parameterizing ``body`` to which
            the values from ``indexset`` will be assigned.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  for_loop │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(self, indexset: Iterable[int], loop_parameter: Union[Parameter, None], body: QuantumCircuit, label: Optional[str]=None):
        num_qubits = body.num_qubits
        num_clbits = body.num_clbits
        super().__init__('for_loop', num_qubits, num_clbits, [indexset, loop_parameter, body], label=label)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        from qiskit.circuit import QuantumCircuit
        indexset, loop_parameter, body = parameters
        if not isinstance(loop_parameter, (Parameter, type(None))):
            raise CircuitError(f'ForLoopOp expects a loop_parameter parameter to be either of type Parameter or None, but received {type(loop_parameter)}.')
        if not isinstance(body, QuantumCircuit):
            raise CircuitError(f'ForLoopOp expects a body parameter to be of type QuantumCircuit, but received {type(body)}.')
        if body.num_qubits != self.num_qubits or body.num_clbits != self.num_clbits:
            raise CircuitError(f'Attempted to assign a body parameter with a num_qubits or num_clbits different than that of the ForLoopOp. ForLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} Supplied body num_qubits/clbits: {body.num_qubits}/{body.num_clbits}.')
        if loop_parameter is not None and loop_parameter not in body.parameters and (loop_parameter.name in (p.name for p in body.parameters)):
            warnings.warn(f'The Parameter provided as a loop_parameter was not found on the loop body and so no binding of the indexset to loop parameter will occur. A different Parameter of the same name ({loop_parameter.name}) was found. If you intended to loop over that Parameter, please use that Parameter instance as the loop_parameter.', stacklevel=2)
        indexset = indexset if isinstance(indexset, range) else tuple(indexset)
        self._params = [indexset, loop_parameter, body]

    @property
    def blocks(self):
        return (self._params[2],)

    def replace_blocks(self, blocks):
        body, = blocks
        return ForLoopOp(self.params[0], self.params[1], body, label=self.label)