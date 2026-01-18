from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _query_commutation(first_op: Operation, first_qargs: List, second_op: Operation, second_qargs: List, _commutation_lib: dict) -> Union[bool, None]:
    """Queries and returns the commutation of a pair of operations from a provided commutation library
    Args:
        first_op: first operation.
        first_qargs: first operation's qubits.
        first_cargs: first operation's clbits.
        second_op: second operation.
        second_qargs: second operation's qubits.
        second_cargs: second operation's clbits.
        _commutation_lib (dict): dictionary of commutation relations
    Return:
        True if first_op and second_op commute, False if they do not commute and
        None if the commutation is not in the library
    """
    commutation = _commutation_lib.get((first_op.name, second_op.name), None)
    if commutation is None or isinstance(commutation, bool):
        return commutation
    if isinstance(commutation, dict):
        commutation_after_placement = commutation.get(_get_relative_placement(first_qargs, second_qargs), None)
        if isinstance(commutation_after_placement, dict):
            return commutation_after_placement.get((_hashable_parameters(first_op.params), _hashable_parameters(second_op.params)), None)
        else:
            return commutation_after_placement
    else:
        raise ValueError('Expected commutation to be None, bool or a dict')