import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
def _condition_op_eq(node1, node2, bit_indices1, bit_indices2):
    cond1 = node1.op.condition
    cond2 = node2.op.condition
    if isinstance(cond1, expr.Expr) and isinstance(cond2, expr.Expr):
        if not expr.structurally_equivalent(cond1, cond2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)):
            return False
    elif isinstance(cond1, expr.Expr) or isinstance(cond2, expr.Expr):
        return False
    elif not _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2):
        return False
    return len(node1.op.blocks) == len(node2.op.blocks) and all((_circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1) == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2) for block1, block2 in zip(node1.op.blocks, node2.op.blocks)))