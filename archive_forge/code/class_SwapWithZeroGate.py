from typing import Sequence, Union, Tuple
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmtp
@attr.frozen
class SwapWithZeroGate(infra.GateWithRegisters):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_{n-1}> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """
    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    def __attrs_post_init__(self):
        assert self.n_target_registers <= 2 ** self.selection_bitsize

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return (infra.SelectionRegister('selection', self.selection_bitsize, self.n_target_registers),)

    @cached_property
    def target_registers(self) -> Tuple[infra.Register, ...]:
        return (infra.Register('target', bitsize=self.target_bitsize, shape=self.n_target_registers),)

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.selection_registers, *self.target_registers])

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        selection, target = (quregs['selection'], quregs['target'])
        assert target.shape == (self.n_target_registers, self.target_bitsize)
        cswap_n = MultiTargetCSwapApprox(self.target_bitsize)
        for j in range(len(selection)):
            for i in range(0, self.n_target_registers - 2 ** j, 2 ** (j + 1)):
                yield cswap_n.on_registers(control=selection[len(selection) - j - 1], target_x=target[i], target_y=target[i + 2 ** j])

    def __repr__(self) -> str:
        return f'cirq_ft.SwapWithZeroGate({self.selection_bitsize},{self.target_bitsize},{self.n_target_registers})'

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@(r⇋0)'] * self.selection_bitsize
        for i in range(self.n_target_registers):
            wire_symbols += [f'swap_{i}'] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)