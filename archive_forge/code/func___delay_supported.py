from __future__ import annotations
from collections.abc import Iterable
import logging
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
def __delay_supported(self, qarg: int) -> bool:
    """Delay operation is supported on the qubit (qarg) or not."""
    if self.target is None or self.target.instruction_supported('delay', qargs=(qarg,)):
        return True
    return False