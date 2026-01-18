from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        