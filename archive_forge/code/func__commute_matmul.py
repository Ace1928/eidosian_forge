from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _commute_matmul(first_ops: Operation, first_qargs: List, second_op: Operation, second_qargs: List):
    qarg = {q: i for i, q in enumerate(first_qargs)}
    num_qubits = len(qarg)
    for q in second_qargs:
        if q not in qarg:
            qarg[q] = num_qubits
            num_qubits += 1
    first_qarg = tuple((qarg[q] for q in first_qargs))
    second_qarg = tuple((qarg[q] for q in second_qargs))
    operator_1 = Operator(first_ops, input_dims=(2,) * len(first_qarg), output_dims=(2,) * len(first_qarg))
    operator_2 = Operator(second_op, input_dims=(2,) * len(second_qarg), output_dims=(2,) * len(second_qarg))
    if first_qarg == second_qarg:
        op12 = operator_1.compose(operator_2)
        op21 = operator_2.compose(operator_1)
    else:
        extra_qarg2 = num_qubits - len(first_qarg)
        if extra_qarg2:
            id_op = _identity_op(extra_qarg2)
            operator_1 = id_op.tensor(operator_1)
        op12 = operator_1.compose(operator_2, qargs=second_qarg, front=False)
        op21 = operator_1.compose(operator_2, qargs=second_qarg, front=True)
    ret = op12 == op21
    return ret