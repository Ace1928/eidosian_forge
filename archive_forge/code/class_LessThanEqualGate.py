from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@deprecated_cirq_ft_class()
@attr.frozen
class LessThanEqualGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|z> = |x>|y> |z ^ (x <= y)>"""
    x_bitsize: int
    y_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return ([2] * self.x_bitsize, [2] * self.y_bitsize, [2])

    def with_registers(self, *new_registers) -> 'LessThanEqualGate':
        return LessThanEqualGate(len(new_registers[0]), len(new_registers[1]))

    def apply(self, *register_vals: int) -> Union[int, int, Iterable[int]]:
        x_val, y_val, target_val = register_vals
        return (x_val, y_val, target_val ^ (x_val <= y_val))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['In(x)'] * self.x_bitsize
        wire_symbols += ['In(y)'] * self.y_bitsize
        wire_symbols += ['+(x <= y)']
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented

    def __repr__(self) -> str:
        return f'cirq_ft.LessThanEqualGate({self.x_bitsize}, {self.y_bitsize})'

    def _decompose_via_tree(self, context: cirq.DecompositionContext, X: Sequence[cirq.Qid], Y: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Returns comparison oracle from https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

        This decomposition follows the tree structure of (FIG. 2)
        """
        if len(X) == 1:
            return
        if len(X) == 2:
            yield BiQubitsMixer().on_registers(x=X, y=Y, ancilla=context.qubit_manager.qalloc(3))
            return
        m = len(X) // 2
        yield self._decompose_via_tree(context, X[:m], Y[:m])
        yield self._decompose_via_tree(context, X[m:], Y[m:])
        yield BiQubitsMixer().on_registers(x=(X[m - 1], X[-1]), y=(Y[m - 1], Y[-1]), ancilla=context.qubit_manager.qalloc(3))

    def _decompose_with_context_(self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext]=None) -> cirq.OP_TREE:
        """Decomposes the gate in a T-complexity optimal way.

        The construction can be broken in 4 parts:
            1. In case of differing bitsizes then a multicontrol And Gate
                - Section III.A. https://arxiv.org/abs/1805.03662) is used to check whether
                the extra prefix is equal to zero:
                    - result stored in: `prefix_equality` qubit.
            2. The tree structure (FIG. 2) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
                followed by a SingleQubitCompare to compute the result of comparison of
                the suffixes of equal length:
                    - result stored in: `less_than` and `greater_than` with equality in qubits[-2]
            3. The results from the previous two steps are combined to update the target qubit.
            4. The adjoint of the previous operations is added to restore the input qubits
                to their original state and clean the ancilla qubits.
        """
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        lhs, rhs, target = (qubits[:self.x_bitsize], qubits[self.x_bitsize:-1], qubits[-1])
        n = min(len(lhs), len(rhs))
        prefix_equality = None
        adjoint: List[cirq.Operation] = []
        if len(lhs) != len(rhs):
            prefix_equality, = context.qubit_manager.qalloc(1)
            if len(lhs) > len(rhs):
                for op in cirq.flatten_to_ops(_equality_with_zero(context, lhs[:-n], prefix_equality)):
                    yield op
                    adjoint.append(cirq.inverse(op))
            else:
                for op in cirq.flatten_to_ops(_equality_with_zero(context, rhs[:-n], prefix_equality)):
                    yield op
                    adjoint.append(cirq.inverse(op))
                yield (cirq.X(target), cirq.CNOT(prefix_equality, target))
        lhs = lhs[-n:]
        rhs = rhs[-n:]
        for op in cirq.flatten_to_ops(self._decompose_via_tree(context, lhs, rhs)):
            yield op
            adjoint.append(cirq.inverse(op))
        less_than, greater_than = context.qubit_manager.qalloc(2)
        yield SingleQubitCompare().on_registers(a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than)
        adjoint.append(SingleQubitCompare(adjoint=True).on_registers(a=lhs[-1], b=rhs[-1], less_than=less_than, greater_than=greater_than))
        if prefix_equality is None:
            yield cirq.X(target)
            yield cirq.CNOT(greater_than, target)
        else:
            less_than_or_equal, = context.qubit_manager.qalloc(1)
            yield and_gate.And([1, 0]).on(prefix_equality, greater_than, less_than_or_equal)
            adjoint.append(and_gate.And([1, 0], adjoint=True).on(prefix_equality, greater_than, less_than_or_equal))
            yield cirq.CNOT(less_than_or_equal, target)
        yield from reversed(adjoint)

    def _t_complexity_(self) -> infra.TComplexity:
        n = min(self.x_bitsize, self.y_bitsize)
        d = max(self.x_bitsize, self.y_bitsize) - n
        is_second_longer = self.y_bitsize > self.x_bitsize
        if d == 0:
            return infra.TComplexity(t=8 * n - 4, clifford=46 * n - 17)
        elif d == 1:
            return infra.TComplexity(t=8 * n, clifford=46 * n + 3 + 2 * is_second_longer)
        else:
            return infra.TComplexity(t=8 * n + 4 * d - 4, clifford=46 * n + 17 * d - 14 + 2 * is_second_longer)

    def _has_unitary_(self):
        return True