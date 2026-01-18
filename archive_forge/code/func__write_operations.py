from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def _write_operations(self, op_tree: 'cirq.OP_TREE', output: Callable[[str], None], output_line_gap: Callable[[int], None]) -> None:

    def keep(op: 'cirq.Operation') -> bool:
        return protocols.qasm(op, args=self.args, default=None) is not None

    def fallback(op):
        if len(op.qubits) not in [1, 2]:
            return NotImplemented
        mat = protocols.unitary(op, None)
        if mat is None:
            return NotImplemented
        if len(op.qubits) == 1:
            return QasmUGate.from_matrix(mat).on(*op.qubits)
        return QasmTwoQubitGate.from_matrix(mat).on(*op.qubits)

    def on_stuck(bad_op):
        return ValueError(f'Cannot output operation as QASM: {bad_op!r}')
    for main_op in ops.flatten_op_tree(op_tree):
        decomposed = protocols.decompose(main_op, keep=keep, fallback_decomposer=fallback, on_stuck_raise=on_stuck)
        qasms = [protocols.qasm(op, args=self.args) for op in decomposed]
        should_annotate = decomposed != [main_op] or qasms[0].count('\n') > 1
        if should_annotate:
            output_line_gap(1)
            if isinstance(main_op, ops.GateOperation):
                x = str(main_op.gate).replace('\n', '\n //')
                output(f'// Gate: {x!s}\n')
            else:
                x = str(main_op).replace('\n', '\n //')
                output(f'// Operation: {x!s}\n')
        for qasm in qasms:
            output(qasm)
        if should_annotate:
            output_line_gap(1)