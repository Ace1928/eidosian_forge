from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _has_measure(circuit: QuantumCircuit) -> bool:
    for instruction in reversed(circuit):
        if isinstance(instruction.operation, Measure):
            return True
        elif isinstance(instruction.operation, ControlFlowOp):
            for block in instruction.operation.blocks:
                if _has_measure(block):
                    return True
    return False