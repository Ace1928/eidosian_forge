from typing import Sequence, Union, Tuple
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmtp
@attr.frozen
class MultiTargetCSwapApprox(MultiTargetCSwap):
    """Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements the unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored, see the reference for more details.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        control, target_x, target_y = (quregs['control'], quregs['target_x'], quregs['target_y'])

        def g(q: cirq.Qid, adjoint=False) -> cirq.ops.op_tree.OpTree:
            yield [cirq.S(q), cirq.H(q)]
            yield (cirq.T(q) ** (1 - 2 * adjoint))
            yield [cirq.H(q), cirq.S(q) ** (-1)]
        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(target_x, target_y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(target_x, target_y)]
        g_inv_on_y = [list(g(q, True)) for q in target_y]
        g_on_y = [list(g(q)) for q in target_y]
        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield mcmtp.MultiTargetCNOT(len(target_y)).on(*control, *target_y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(('@(approx)',) + ('swap_x',) * self.bitsize + ('swap_y',) * self.bitsize)
        return cirq.CircuitDiagramInfo(('@(approx)',) + ('×(x)',) * self.bitsize + ('×(y)',) * self.bitsize)

    def __repr__(self) -> str:
        return f'cirq_ft.MultiTargetCSwapApprox({self.bitsize})'

    def _t_complexity_(self) -> infra.TComplexity:
        """TComplexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        return infra.TComplexity(t=4 * n, clifford=22 * n - 1)