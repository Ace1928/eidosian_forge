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
class AdditionGate(cirq.ArithmeticGate):
    """Applies U|p>|q> -> |p>|p+q>.

    Args:
        bitsize: The number of bits used to represent each integer p and q.
            Note that this adder does not detect overflow if bitsize is not
            large enough to hold p + q and simply drops the most significant bits.

    References:
        [Halving the cost of quantum addition](https://arxiv.org/abs/1709.06648)
    """
    bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return ([2] * self.bitsize, [2] * self.bitsize)

    def with_registers(self, *new_registers) -> 'AdditionGate':
        return AdditionGate(len(new_registers[0]))

    def apply(self, *register_values: int) -> Union[int, Iterable[int]]:
        p, q = register_values
        return (p, p + q)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['In(x)'] * self.bitsize
        wire_symbols += ['In(y)/Out(x+y)'] * self.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self):
        return True

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.bitsize - 1:
            return
        else:
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(anc[depth - 1], out[depth])
            yield and_gate.And().on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield cirq.CX(anc[depth - 1], anc[depth])
            yield and_gate.And(adjoint=True).on(inp[depth], out[depth], anc[depth])
            yield cirq.CX(anc[depth - 1], inp[depth])
            yield cirq.CX(inp[depth], out[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def _decompose_with_context_(self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext]=None) -> cirq.OP_TREE:
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        input_bits = qubits[:self.bitsize]
        output_bits = qubits[self.bitsize:]
        ancillas = context.qubit_manager.qalloc(self.bitsize - 1)
        yield and_gate.And().on(input_bits[0], output_bits[0], ancillas[0])
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        yield cirq.CX(ancillas[-1], output_bits[-1])
        yield cirq.CX(input_bits[-1], output_bits[-1])
        yield from self._right_building_block(input_bits, output_bits, ancillas, self.bitsize - 2)
        yield and_gate.And(adjoint=True).on(input_bits[0], output_bits[0], ancillas[0])
        yield cirq.CX(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

    def _t_complexity_(self) -> infra.TComplexity:
        num_clifford = (self.bitsize - 2) * 19 + 16
        return infra.TComplexity(t=4 * self.bitsize - 4, clifford=num_clifford)

    def __repr__(self) -> str:
        return f'cirq_ft.AdditionGate({self.bitsize})'