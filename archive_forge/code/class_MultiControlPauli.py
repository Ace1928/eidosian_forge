from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
@attr.frozen
class MultiControlPauli(infra.GateWithRegisters):
    """Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-1$
    clean ancillas using a multi-controlled `AND` gate.

    References:
        [Constructing Large Controlled Nots]
        (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)
    """
    cvs: Tuple[int, ...] = attr.field(converter=lambda v: (v,) if isinstance(v, int) else tuple(v))
    target_gate: cirq.Pauli = cirq.X

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature.build(controls=len(self.cvs), target=1)

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray['cirq.Qid']) -> cirq.OP_TREE:
        controls, target = (quregs['controls'], quregs['target'])
        qm = context.qubit_manager
        and_ancilla, and_target = (np.array(qm.qalloc(len(self.cvs) - 2)), qm.qalloc(1))
        yield and_gate.And(self.cvs).on_registers(ctrl=controls[:, np.newaxis], junk=and_ancilla[:, np.newaxis], target=and_target)
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_gate.And(self.cvs, adjoint=True).on_registers(ctrl=controls[:, np.newaxis], junk=and_ancilla[:, np.newaxis], target=and_target)
        qm.qfree([*and_ancilla, *and_target])

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if b else '@(0)' for b in self.cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> infra.TComplexity:
        and_cost = infra.t_complexity(and_gate.And(self.cvs))
        controlled_pauli_cost = infra.t_complexity(self.target_gate.controlled(1))
        and_inv_cost = infra.t_complexity(and_gate.And(self.cvs, adjoint=True))
        return and_cost + controlled_pauli_cost + and_inv_cost

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        return cirq.apply_unitary(self.target_gate.controlled(control_values=self.cvs), args)

    def _has_unitary_(self) -> bool:
        return True