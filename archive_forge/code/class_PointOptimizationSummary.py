import abc
from collections import defaultdict
from typing import Dict, Callable, Iterable, Optional, Sequence, TYPE_CHECKING, Tuple, cast
from cirq import ops
class PointOptimizationSummary:
    """A description of a local optimization to perform."""

    def __init__(self, clear_span: int, clear_qubits: Iterable['cirq.Qid'], new_operations: 'cirq.OP_TREE', preserve_moments: bool=False) -> None:
        """Inits PointOptimizationSummary.

        Args:
            clear_span: Defines the range of moments to affect. Specifically,
                refers to the indices in range(start, start+clear_span) where
                start is an index known from surrounding context.
            clear_qubits: Defines the set of qubits that should be cleared
                with each affected moment.
            new_operations: The operations to replace the cleared out
                operations with.
            preserve_moments: If set, `cirq.Moment` instances within
                `new_operations` will be preserved exactly. Normally the
                operations would be repacked to fit better into the
                target space, which may move them between moments.
                Please be advised that a PointOptimizer consuming this
                summary will flatten operations no matter what,
                see https://github.com/quantumlib/Cirq/issues/2406.
        """
        self.new_operations = tuple(ops.flatten_op_tree(new_operations, preserve_moments=preserve_moments))
        self.clear_span = clear_span
        self.clear_qubits = tuple(clear_qubits)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.clear_span == other.clear_span and self.clear_qubits == other.clear_qubits and (self.new_operations == other.new_operations)

    def __ne__(self, other):
        return not self == other

    def __hash__(self) -> int:
        return hash((PointOptimizationSummary, self.clear_span, self.clear_qubits, self.new_operations))

    def __repr__(self) -> str:
        return f'cirq.PointOptimizationSummary({self.clear_span!r}, {self.clear_qubits!r}, {self.new_operations!r})'